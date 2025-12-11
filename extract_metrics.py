#!/usr/bin/env python
"""
Extract Real Metrics from EETA Training

This script:
1. Loads your trained checkpoints (ep49, ep79, ep149, etc.)
2. Runs them through the dataset to collect REAL metrics
3. Generates visualizations from REAL data

Usage:
    python extract_metrics.py
    python extract_metrics.py --checkpoint-dir experiments/checkpoints/20251210_110803/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from datetime import datetime
from collections import defaultdict

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def find_checkpoints(checkpoint_dir: str = None) -> dict:
    """Find all available checkpoints."""
    if checkpoint_dir:
        base_dir = Path(checkpoint_dir)
    else:
        # Find most recent checkpoint directory
        checkpoint_base = Path("experiments/checkpoints")
        if not checkpoint_base.exists():
            return {}
        dirs = sorted(checkpoint_base.iterdir(), reverse=True)
        if not dirs:
            return {}
        base_dir = dirs[0]
    
    checkpoints = {}
    for f in base_dir.glob("*.pt"):
        name = f.stem
        if "ep" in name:
            # Extract episode number
            ep_num = int(name.split("ep")[1].split("_")[0])
            checkpoints[ep_num] = f
        elif name == "final_model":
            checkpoints["final"] = f
        elif name == "best_model":
            checkpoints["best"] = f
        elif "best" in name:
            checkpoints["best"] = f
    
    return checkpoints


def load_agent(checkpoint_path: str, config: dict):
    """Load agent from checkpoint."""
    from src.rl.dqn import DQNAgent
    
    dqn_config = config.get('dqn', {})
    
    agent = DQNAgent(
        state_dim=dqn_config.get('state_dim', 43),
        action_dim=dqn_config.get('action_dim', 5),
        hidden_dims=dqn_config.get('hidden_dims', [128, 64]),
        learning_rate=dqn_config.get('learning_rate', 0.001),
        gamma=dqn_config.get('gamma', 0.99),
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1.0,
        batch_size=dqn_config.get('batch_size', 32),
        buffer_size=dqn_config.get('replay_buffer_size', 10000)
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    agent.q_network.load_state_dict(checkpoint['q_network'])
    agent.epsilon = checkpoint.get('epsilon', 0.0)
    
    return agent, checkpoint


def evaluate_checkpoint(agent, dataset, preprocessor, n_samples: int = None):
    """Evaluate a checkpoint and collect detailed metrics."""
    from src.environment.trading_env import EarningsTradingEnv
    from src.rl.thompson import ThompsonSampler
    
    env = EarningsTradingEnv(
        data=dataset if n_samples is None else dataset.head(n_samples),
        preprocessor=preprocessor,
        shuffle=False
    )
    
    thompson = ThompsonSampler()
    
    state = env.reset()
    
    # Collect data
    actions = []
    rewards = []
    pnls = []
    q_values_list = []
    thompson_selections = []
    
    while not env.done:
        # Get Q-values
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = agent.q_network(state_tensor).numpy()[0]
        
        q_values_list.append({
            'max_q': float(np.max(q_values)),
            'min_q': float(np.min(q_values)),
            'mean_q': float(np.mean(q_values)),
            'q_values': [float(q) for q in q_values]
        })
        
        # Select action (greedy)
        action = int(np.argmax(q_values))
        actions.append(action)
        
        # Thompson sampling for position size
        bucket_idx, position_size = thompson.select_size(
            confidence=float(np.max(q_values)),
            volatility=0.3
        )
        thompson_selections.append({
            'bucket_idx': int(bucket_idx),
            'position_size': float(position_size)
        })
        
        # Step
        next_state, reward, done, info = env.step(action)
        rewards.append(float(reward))
        pnl = info.get('pnl_pct', 0.0)
        pnls.append(float(pnl))
        
        # Update Thompson
        if action != 0:
            thompson.update(bucket_idx, pnl > 0)
        
        state = next_state
    
    # Compute metrics
    action_counts = {i: actions.count(i) for i in range(5)}
    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / max(len(pnls), 1)
    
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls))
    else:
        sharpe = 0.0
    
    return {
        'actions': actions,
        'action_counts': action_counts,
        'rewards': rewards,
        'pnls': pnls,
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'q_values': q_values_list,
        'thompson_selections': thompson_selections,
        'thompson_alphas': [float(a) for a in thompson.alphas],
        'thompson_betas': [float(b) for b in thompson.betas],
        'n_trades': len([a for a in actions if a != 0])
    }


def extract_all_metrics(checkpoint_dir: str = None):
    """Extract metrics from all checkpoints."""
    from src.utils.config import get_config
    from src.data.preprocessor import StatePreprocessor
    
    print("=" * 60)
    print("EXTRACTING REAL METRICS FROM CHECKPOINTS")
    print("=" * 60)
    
    # Load dataset
    data_path = Path("data/processed/earnings_dataset.parquet")
    if not data_path.exists():
        print("❌ Dataset not found!")
        return None
    
    dataset = pd.read_parquet(data_path)
    print(f"✓ Loaded {len(dataset)} events")
    
    # Setup
    config = get_config()
    preprocessor = StatePreprocessor()
    preprocessor.fit(dataset)
    
    # Find checkpoints
    checkpoints = find_checkpoints(checkpoint_dir)
    if not checkpoints:
        print("❌ No checkpoints found!")
        return None
    
    print(f"✓ Found {len(checkpoints)} checkpoints: {list(checkpoints.keys())}")
    
    # Extract metrics from each checkpoint
    all_metrics = {}
    checkpoint_info = []
    
    for ep_key, ckpt_path in sorted(checkpoints.items(), key=lambda x: str(x[0])):
        print(f"\n📊 Processing checkpoint: {ep_key}")
        
        agent, ckpt_data = load_agent(str(ckpt_path), config._raw)
        
        # Get epsilon from checkpoint
        epsilon = ckpt_data.get('epsilon', 0.0)
        training_steps = ckpt_data.get('training_steps', 0)
        losses = ckpt_data.get('losses', [])
        
        # Evaluate
        metrics = evaluate_checkpoint(agent, dataset, preprocessor)
        
        # Store
        ep_num = ep_key if isinstance(ep_key, int) else 999
        all_metrics[ep_key] = {
            'episode': ep_num,
            'epsilon': epsilon,
            'training_steps': training_steps,
            'recent_losses': losses[-100:] if losses else [],
            **metrics
        }
        
        checkpoint_info.append({
            'episode': ep_num,
            'epsilon': epsilon,
            'total_pnl': metrics['total_pnl'],
            'win_rate': metrics['win_rate'],
            'sharpe': metrics['sharpe'],
            'n_trades': metrics['n_trades'],
            'action_0': metrics['action_counts'].get(0, 0),
            'action_1': metrics['action_counts'].get(1, 0),
            'action_2': metrics['action_counts'].get(2, 0),
            'action_3': metrics['action_counts'].get(3, 0),
            'action_4': metrics['action_counts'].get(4, 0),
        })
        
        print(f"   P&L: {metrics['total_pnl']*100:.2f}% | "
              f"Win: {metrics['win_rate']*100:.1f}% | "
              f"Sharpe: {metrics['sharpe']:.2f} | "
              f"ε: {epsilon:.3f}")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(checkpoint_info)
    summary_df = summary_df.sort_values('episode')
    
    return {
        'checkpoints': all_metrics,
        'summary': summary_df
    }


def create_real_dqn_visualizations(metrics: dict) -> dict:
    """Create DQN visualizations from real data."""
    summary = metrics['summary']
    checkpoints = metrics['checkpoints']
    
    figures = {}
    
    # 1. Epsilon Decay (Real)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=summary['episode'],
        y=summary['epsilon'],
        mode='lines+markers',
        name='Epsilon',
        line=dict(color='#ff9800', width=3),
        marker=dict(size=10)
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Full Exploration")
    fig.add_hline(y=0.05, line_dash="dash", line_color="gray", annotation_text="Min Epsilon")
    fig.update_layout(
        title="<b>Epsilon Decay (REAL DATA)</b><br><sup>Exploration → Exploitation transition across checkpoints</sup>",
        xaxis_title="Episode",
        yaxis_title="Epsilon",
        height=400
    )
    figures['epsilon_decay'] = fig
    
    # 2. Q-Value Distribution per Checkpoint
    fig = go.Figure()
    colors = ['#e91e63', '#9c27b0', '#2196f3', '#4caf50', '#ff9800']
    
    for i, (ep_key, data) in enumerate(checkpoints.items()):
        if isinstance(ep_key, int):
            q_means = [q['mean_q'] for q in data['q_values']]
            fig.add_trace(go.Box(
                y=q_means,
                name=f'Ep {ep_key}',
                marker_color=colors[i % len(colors)]
            ))
    
    fig.update_layout(
        title="<b>Q-Value Distribution (REAL DATA)</b><br><sup>How Q-values evolve across training</sup>",
        yaxis_title="Mean Q-Value",
        height=400
    )
    figures['q_value_distribution'] = fig
    
    # 3. Q-Values by Action (for best checkpoint)
    best_key = max([k for k in checkpoints.keys() if isinstance(k, int)])
    best_data = checkpoints[best_key]
    
    # Aggregate Q-values by chosen action
    action_q_values = defaultdict(list)
    for q_data, action in zip(best_data['q_values'], best_data['actions']):
        action_q_values[action].append(q_data['q_values'][action])
    
    action_names = ['NO_TRADE', 'LONG', 'SHORT', 'LONG_VOL', 'SHORT_VOL']
    action_colors = ['#9e9e9e', '#00c853', '#f44336', '#2196f3', '#ff9800']
    
    fig = go.Figure()
    for action in range(5):
        if action_q_values[action]:
            fig.add_trace(go.Box(
                y=action_q_values[action],
                name=action_names[action],
                marker_color=action_colors[action]
            ))
    
    fig.update_layout(
        title=f"<b>Q-Values by Action (REAL DATA - Ep {best_key})</b><br><sup>Which actions have highest estimated value</sup>",
        yaxis_title="Q-Value",
        height=400
    )
    figures['q_values_by_action'] = fig
    
    # 4. Loss Curve (if available)
    has_losses = False
    for data in checkpoints.values():
        if data.get('recent_losses'):
            has_losses = True
            break
    
    if has_losses:
        fig = go.Figure()
        for ep_key, data in checkpoints.items():
            if data.get('recent_losses') and isinstance(ep_key, int):
                losses = data['recent_losses']
                fig.add_trace(go.Scatter(
                    y=losses,
                    mode='lines',
                    name=f'Ep {ep_key}',
                    opacity=0.7
                ))
        
        fig.update_layout(
            title="<b>DQN Loss (REAL DATA)</b><br><sup>Training loss from checkpoints</sup>",
            xaxis_title="Training Step",
            yaxis_title="Loss",
            height=400
        )
        figures['loss_curve'] = fig
    
    return figures


def create_real_thompson_visualizations(metrics: dict) -> dict:
    """Create Thompson Sampling visualizations from real data."""
    checkpoints = metrics['checkpoints']
    summary = metrics['summary']
    
    figures = {}
    
    bucket_names = ['0.5%', '1%', '2%', '3%', '5%']
    colors = ['#e91e63', '#9c27b0', '#2196f3', '#4caf50', '#ff9800']
    
    # 1. Beta Parameters Evolution
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Alpha (Successes)", "Beta (Failures)"))
    
    episodes = []
    alphas_by_bucket = {i: [] for i in range(5)}
    betas_by_bucket = {i: [] for i in range(5)}
    
    for ep_key, data in sorted(checkpoints.items(), key=lambda x: str(x[0])):
        if isinstance(ep_key, int) and data.get('thompson_alphas'):
            episodes.append(ep_key)
            for i, (a, b) in enumerate(zip(data['thompson_alphas'], data['thompson_betas'])):
                alphas_by_bucket[i].append(a)
                betas_by_bucket[i].append(b)
    
    for i, (name, color) in enumerate(zip(bucket_names, colors)):
        fig.add_trace(go.Scatter(
            x=episodes, y=alphas_by_bucket[i],
            mode='lines+markers', name=name,
            line=dict(color=color, width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=episodes, y=betas_by_bucket[i],
            mode='lines+markers', name=name,
            line=dict(color=color, width=2, dash='dash'),
            showlegend=False
        ), row=1, col=2)
    
    fig.update_layout(
        title="<b>Thompson Sampling Beta Parameters (REAL DATA)</b><br><sup>Bayesian learning of position size performance</sup>",
        height=400
    )
    figures['thompson_beta_evolution'] = fig
    
    # 2. Win Rate by Bucket (Final)
    best_key = max([k for k in checkpoints.keys() if isinstance(k, int)])
    best_data = checkpoints[best_key]
    
    if best_data.get('thompson_alphas'):
        win_rates = []
        for a, b in zip(best_data['thompson_alphas'], best_data['thompson_betas']):
            win_rates.append(a / (a + b) * 100 if (a + b) > 0 else 50)
        
        fig = go.Figure(data=[go.Bar(
            x=bucket_names,
            y=win_rates,
            marker_color=colors,
            text=[f'{wr:.1f}%' for wr in win_rates],
            textposition='outside'
        )])
        fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Random (50%)")
        fig.update_layout(
            title=f"<b>Win Rate by Position Size (REAL DATA - Ep {best_key})</b><br><sup>Which position sizes are most profitable</sup>",
            xaxis_title="Position Size",
            yaxis_title="Win Rate (%)",
            height=400
        )
        figures['thompson_win_rates'] = fig
    
    # 3. Selection Frequency
    bucket_counts = {i: 0 for i in range(5)}
    for data in checkpoints.values():
        for sel in data.get('thompson_selections', []):
            bucket_counts[sel['bucket_idx']] += 1
    
    fig = go.Figure(data=[go.Pie(
        labels=bucket_names,
        values=list(bucket_counts.values()),
        marker_colors=colors,
        textinfo='percent+label',
        hole=0.3
    )])
    fig.update_layout(
        title="<b>Position Size Selection Frequency (REAL DATA)</b><br><sup>How often each size was chosen</sup>",
        height=400
    )
    figures['thompson_selection_freq'] = fig
    
    return figures


def create_real_training_visualizations(metrics: dict) -> dict:
    """Create training progress visualizations from real data."""
    summary = metrics['summary']
    checkpoints = metrics['checkpoints']
    
    figures = {}
    
    # 1. Performance Progression
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("P&L Progression", "Win Rate", "Sharpe Ratio", "Action Distribution")
    )
    
    # P&L
    fig.add_trace(go.Scatter(
        x=summary['episode'], y=summary['total_pnl'] * 100,
        mode='lines+markers', name='P&L',
        line=dict(color='#00c853', width=2),
        marker=dict(size=8)
    ), row=1, col=1)
    
    # Win Rate
    fig.add_trace(go.Scatter(
        x=summary['episode'], y=summary['win_rate'] * 100,
        mode='lines+markers', name='Win Rate',
        line=dict(color='#9c27b0', width=2),
        marker=dict(size=8)
    ), row=1, col=2)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=1, col=2)
    
    # Sharpe
    fig.add_trace(go.Scatter(
        x=summary['episode'], y=summary['sharpe'],
        mode='lines+markers', name='Sharpe',
        line=dict(color='#2196f3', width=2),
        marker=dict(size=8)
    ), row=2, col=1)
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Action Distribution (stacked bar)
    action_names = ['NO_TRADE', 'LONG', 'SHORT', 'LONG_VOL', 'SHORT_VOL']
    action_colors = ['#9e9e9e', '#00c853', '#f44336', '#2196f3', '#ff9800']
    
    for i, (name, color) in enumerate(zip(action_names, action_colors)):
        fig.add_trace(go.Bar(
            x=summary['episode'],
            y=summary[f'action_{i}'],
            name=name,
            marker_color=color
        ), row=2, col=2)
    
    fig.update_layout(
        title="<b>Training Progress (REAL DATA)</b><br><sup>Performance metrics across checkpoints</sup>",
        height=600,
        barmode='stack'
    )
    figures['training_progress'] = fig
    
    # 2. Action Distribution Evolution
    fig = go.Figure()
    for i, (name, color) in enumerate(zip(action_names, action_colors)):
        fig.add_trace(go.Scatter(
            x=summary['episode'],
            y=summary[f'action_{i}'],
            mode='lines',
            name=name,
            stackgroup='one',
            line=dict(color=color)
        ))
    
    fig.update_layout(
        title="<b>Action Distribution Over Training (REAL DATA)</b><br><sup>How strategy evolved</sup>",
        xaxis_title="Episode",
        yaxis_title="Action Count",
        height=400
    )
    figures['action_evolution'] = fig
    
    return figures


def create_rl_comparison_dashboard(metrics: dict) -> go.Figure:
    """Create comprehensive RL comparison dashboard."""
    summary = metrics['summary']
    checkpoints = metrics['checkpoints']
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "DQN: Epsilon Decay",
            "DQN: Q-Value Growth",
            "Thompson: Win Rates",
            "Performance: P&L",
            "Performance: Sharpe",
            "Strategy: Actions"
        ),
        vertical_spacing=0.12,
        # --- THE FIX IS HERE ---
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "domain"}] 
        ]
    # fig = make_subplots(
    #     rows=2, cols=3,
    #     subplot_titles=(
    #         "DQN: Epsilon Decay",
    #         "DQN: Q-Value Growth",
    #         "Thompson: Win Rates",
    #         "Performance: P&L",
    #         "Performance: Sharpe",
    #         "Strategy: Actions"
    #     ),
    #     vertical_spacing=0.12
    )
    
    bucket_names = ['0.5%', '1%', '2%', '3%', '5%']
    colors = ['#e91e63', '#9c27b0', '#2196f3', '#4caf50', '#ff9800']
    
    # 1. Epsilon
    fig.add_trace(go.Scatter(
        x=summary['episode'], y=summary['epsilon'],
        mode='lines+markers', line=dict(color='#ff9800', width=2)
    ), row=1, col=1)
    
    # 2. Q-Values
    episodes = []
    mean_q = []
    for ep_key, data in sorted(checkpoints.items(), key=lambda x: str(x[0])):
        if isinstance(ep_key, int):
            episodes.append(ep_key)
            q_means = [q['mean_q'] for q in data['q_values']]
            mean_q.append(np.mean(q_means))
    
    fig.add_trace(go.Scatter(
        x=episodes, y=mean_q,
        mode='lines+markers', line=dict(color='#2196f3', width=2)
    ), row=1, col=2)
    
    # 3. Thompson Win Rates
    for ep_key, data in sorted(checkpoints.items(), key=lambda x: str(x[0])):
        if isinstance(ep_key, int) and data.get('thompson_alphas'):
            win_rates = [a/(a+b)*100 for a, b in zip(data['thompson_alphas'], data['thompson_betas'])]
            fig.add_trace(go.Bar(
                x=bucket_names, y=win_rates,
                name=f'Ep {ep_key}',
                marker_color=[f'rgba({int(c[1:3], 16)}, {int(c[3:5], 16)}, {int(c[5:7], 16)}, 0.5)' for c in colors]
            ), row=1, col=3)
            break  # Just show latest
    
    # 4. P&L
    fig.add_trace(go.Scatter(
        x=summary['episode'], y=summary['total_pnl'] * 100,
        mode='lines+markers', line=dict(color='#00c853', width=2),
        fill='tozeroy', fillcolor='rgba(0, 200, 83, 0.1)'
    ), row=2, col=1)
    
    # 5. Sharpe
    fig.add_trace(go.Scatter(
        x=summary['episode'], y=summary['sharpe'],
        mode='lines+markers', line=dict(color='#9c27b0', width=2)
    ), row=2, col=2)
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=2)
    
    # 6. Actions (final checkpoint)
    action_names = ['NO_TRADE', 'LONG', 'SHORT', 'LONG_VOL', 'SHORT_VOL']
    action_colors = ['#9e9e9e', '#00c853', '#f44336', '#2196f3', '#ff9800']
    last = summary.iloc[-1]
    values = [last[f'action_{i}'] for i in range(5)]
    
    fig.add_trace(go.Pie(
        labels=action_names, values=values,
        marker_colors=action_colors,
        textinfo='percent',
        domain=dict(x=[0.7, 1.0], y=[0, 0.4])
    ), row=2, col=3)
    
    fig.update_layout(
        title="<b>RL Methods Comparison (REAL DATA)</b><br><sup>DQN (action selection) + Thompson Sampling (position sizing)</sup>",
        height=650,
        showlegend=False
    )
    
    return fig


def save_visualizations(figures: dict, output_dir: str):
    """Save all figures as HTML."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, fig in figures.items():
        fig.write_html(output_dir / f"{name}.html")
        print(f"   ✓ {name}.html")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Extract Real EETA Metrics')
    parser.add_argument('--checkpoint-dir', default=None, help='Checkpoint directory')
    parser.add_argument('--output-dir', default='experiments/visualizations/real_data', help='Output directory')
    
    args = parser.parse_args()
    
    # Extract metrics
    metrics = extract_all_metrics(args.checkpoint_dir)
    
    if metrics is None:
        return
    
    # Save raw metrics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics['summary'].to_csv(output_dir / "checkpoint_summary.csv", index=False)
    print(f"\n✓ Summary saved to {output_dir / 'checkpoint_summary.csv'}")
    
    # Save detailed metrics as JSON
    # Convert to serializable format
    serializable = {}
    for key, data in metrics['checkpoints'].items():
        serializable[str(key)] = {
            k: v for k, v in data.items() 
            if k not in ['q_values', 'thompson_selections']  # Too large
        }
    
    with open(output_dir / "checkpoint_metrics.json", 'w') as f:
        json.dump(serializable, f, indent=2)
    
    # Generate visualizations
    print("\n📊 Generating visualizations from REAL data...")
    
    all_figures = {}
    
    # DQN visualizations
    dqn_figs = create_real_dqn_visualizations(metrics)
    all_figures.update(dqn_figs)
    
    # Thompson visualizations
    thompson_figs = create_real_thompson_visualizations(metrics)
    all_figures.update(thompson_figs)
    
    # Training visualizations
    training_figs = create_real_training_visualizations(metrics)
    all_figures.update(training_figs)
    
    # Combined dashboard
    all_figures['rl_comparison_dashboard'] = create_rl_comparison_dashboard(metrics)
    
    # Save all
    save_visualizations(all_figures, args.output_dir)
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print("\n📂 Generated files:")
    for name in all_figures.keys():
        print(f"   - {name}.html")
    
    print(f"\n🌐 Open in browser:")
    print(f"   file://{output_dir.absolute()}/rl_comparison_dashboard.html")


if __name__ == '__main__':
    main()
