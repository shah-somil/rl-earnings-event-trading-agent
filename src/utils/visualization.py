"""
Visualization Module for EETA.

Provides plotting functions for:
- Training dashboard (9-panel analysis)
- Benchmark comparison charts
- Ablation study visualizations
- Performance metrics plots
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed - visualization disabled")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def check_plotting_available():
    """Check if plotting is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for visualization. Install with: pip install matplotlib")
    return True


def create_training_dashboard(
    metrics: Dict[str, Any],
    save_path: str = None,
    figsize: Tuple[int, int] = (15, 12)
) -> Optional['plt.Figure']:
    """
    Create comprehensive 9-panel training analysis dashboard.
    
    Panels:
    1. Training Rewards
    2. Cumulative P&L
    3. Rolling Win Rate
    4. Action Distribution
    5. Thompson Sampling Evolution
    6. DQN Loss
    7. Epsilon Decay
    8. Curriculum Progression
    9. Returns by Difficulty
    
    Args:
        metrics: Dictionary with training metrics
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    check_plotting_available()
    
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle("Earnings Trading Agent - Training Analysis", 
                 fontsize=14, fontweight='bold')
    
    episodes = metrics.get("episodes", list(range(len(metrics.get("rewards", [])))))
    
    # Panel 1: Training Rewards
    ax = axes[0, 0]
    rewards = metrics.get("rewards", [])
    if rewards:
        ax.plot(episodes[:len(rewards)], rewards, alpha=0.5, color="blue", label="Raw")
        # Smoothed rewards
        if len(rewards) > 10:
            smoothed = _smooth(rewards, window=10)
            ax.plot(episodes[:len(smoothed)], smoothed, color="blue", linewidth=2, label="Smoothed")
        # Trend line
        if len(rewards) > 1:
            z = np.polyfit(range(len(rewards)), rewards, 1)
            ax.plot(episodes[:len(rewards)], np.polyval(z, range(len(rewards))),
                    'r--', label=f"Trend: {z[0]:.4f}")
    ax.set_title("Training Rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Cumulative P&L
    ax = axes[0, 1]
    pnl = metrics.get("pnl", metrics.get("cumulative_pnl", []))
    if pnl:
        cumulative = np.cumsum(pnl) if not metrics.get("is_cumulative", False) else pnl
        ax.plot(episodes[:len(cumulative)], cumulative, color="green", linewidth=2)
        ax.fill_between(episodes[:len(cumulative)], 0, cumulative, alpha=0.3, color="green")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_title("Cumulative P&L")
    ax.set_xlabel("Episode")
    ax.set_ylabel("P&L")
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Win Rate (Rolling)
    ax = axes[0, 2]
    win_rates = metrics.get("win_rates", metrics.get("rolling_win_rate", []))
    if win_rates:
        ax.plot(episodes[:len(win_rates)], win_rates, color="purple", linewidth=2)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="50%")
    ax.set_title("Rolling Win Rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate")
    ax.set_ylim([0.3, 0.8])
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Action Distribution
    ax = axes[1, 0]
    action_counts = metrics.get("action_counts", metrics.get("action_distribution", {}))
    if action_counts:
        if isinstance(action_counts, dict):
            actions = list(action_counts.keys())
            counts = list(action_counts.values())
        else:
            actions = ["No Trade", "Long", "Short", "Long Vol", "Short Vol"]
            counts = action_counts[:5] if len(action_counts) >= 5 else action_counts
        colors = ["gray", "green", "red", "blue", "orange"][:len(actions)]
        ax.bar(actions, counts, color=colors)
    ax.set_title("Action Distribution")
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel("Count")
    
    # Panel 5: Thompson Sampling Evolution
    ax = axes[1, 1]
    thompson_data = metrics.get("thompson_expected", metrics.get("thompson", {}))
    if isinstance(thompson_data, dict):
        for size_label, values in thompson_data.items():
            ax.plot(values, label=size_label, alpha=0.8)
    elif isinstance(thompson_data, np.ndarray) and thompson_data.ndim == 2:
        for i, size in enumerate(["0.5%", "1%", "2%", "3%", "5%"]):
            if i < thompson_data.shape[1]:
                ax.plot(thompson_data[:, i], label=size, alpha=0.8)
    ax.set_title("Position Size Learning (Thompson)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Expected Success Rate")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    
    # Panel 6: DQN Loss
    ax = axes[1, 2]
    losses = metrics.get("losses", metrics.get("dqn_loss", []))
    if losses:
        ax.plot(losses, alpha=0.5, color="red")
        if len(losses) > 10:
            smoothed_loss = _smooth(losses, window=10)
            ax.plot(smoothed_loss, color="red", linewidth=2)
    ax.set_title("DQN Loss")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    if losses and max(losses) > 0:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    
    # Panel 7: Epsilon Decay
    ax = axes[2, 0]
    epsilons = metrics.get("epsilons", metrics.get("epsilon", []))
    if epsilons:
        ax.plot(episodes[:len(epsilons)], epsilons, color="orange", linewidth=2)
    ax.set_title("Exploration Rate (ε)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.grid(True, alpha=0.3)
    
    # Panel 8: Curriculum Progression
    ax = axes[2, 1]
    difficulties = metrics.get("difficulties", metrics.get("curriculum", []))
    if difficulties:
        difficulty_map = {"easy": 0, "medium": 1, "hard": 2}
        numeric = [difficulty_map.get(d, 1) if isinstance(d, str) else d for d in difficulties]
        ax.step(episodes[:len(numeric)], numeric, where="post", color="teal", linewidth=2)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Easy", "Medium", "Hard"])
    ax.set_title("Curriculum Progression")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)
    
    # Panel 9: Returns by Difficulty
    ax = axes[2, 2]
    returns_by_diff = metrics.get("returns_by_difficulty", {})
    if returns_by_diff:
        data = [returns_by_diff.get(d, []) for d in ["easy", "medium", "hard"]]
        data = [d for d in data if d]  # Remove empty lists
        if data:
            bp = ax.boxplot(data, labels=["Easy", "Medium", "Hard"][:len(data)], patch_artist=True)
            colors = ["lightgreen", "lightyellow", "lightcoral"][:len(data)]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("Returns by Difficulty")
    ax.set_ylabel("Return (%)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Dashboard saved to {save_path}")
    
    return fig


def plot_benchmark_comparison(
    agent_returns: List[float],
    spy_returns: List[float],
    dates: List = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Optional['plt.Figure']:
    """
    Create benchmark comparison chart (Agent vs SPY).
    
    Args:
        agent_returns: Agent's returns
        spy_returns: SPY returns
        dates: Optional date index
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    check_plotting_available()
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, 
                             gridspec_kw={'height_ratios': [3, 1]})
    
    if dates is None:
        dates = list(range(len(agent_returns)))
    
    # Panel 1: Cumulative Returns
    ax1 = axes[0]
    
    agent_cum = np.cumprod([1 + r for r in agent_returns])
    spy_cum = np.cumprod([1 + r for r in spy_returns[:len(agent_returns)]])
    
    ax1.plot(dates[:len(agent_cum)], agent_cum, label="EETA Agent", 
             color="blue", linewidth=2)
    ax1.plot(dates[:len(spy_cum)], spy_cum, label="Buy & Hold SPY", 
             color="gray", linewidth=2, linestyle="--")
    
    # Fill outperformance/underperformance
    ax1.fill_between(dates[:len(agent_cum)], 1, agent_cum, 
                     where=agent_cum > spy_cum[:len(agent_cum)], 
                     alpha=0.3, color="green", label="Outperformance")
    ax1.fill_between(dates[:len(agent_cum)], 1, agent_cum, 
                     where=agent_cum < spy_cum[:len(agent_cum)], 
                     alpha=0.3, color="red", label="Underperformance")
    
    ax1.set_ylabel("Growth of $1")
    ax1.set_title("EETA Agent vs Buy & Hold SPY")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # Add metrics annotation
    from .metrics import calculate_sharpe_ratio, calculate_max_drawdown
    agent_sharpe = calculate_sharpe_ratio(agent_returns)
    spy_sharpe = calculate_sharpe_ratio(spy_returns[:len(agent_returns)])
    agent_dd = calculate_max_drawdown(agent_returns)
    spy_dd = calculate_max_drawdown(spy_returns[:len(agent_returns)])
    
    metrics_text = (
        f"Agent: Sharpe={agent_sharpe:.2f}, MaxDD={agent_dd:.1%}\n"
        f"SPY:   Sharpe={spy_sharpe:.2f}, MaxDD={spy_dd:.1%}"
    )
    ax1.annotate(metrics_text, xy=(0.02, 0.98), xycoords='axes fraction',
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel 2: Drawdown
    ax2 = axes[1]
    
    agent_dd_series = agent_cum / np.maximum.accumulate(agent_cum) - 1
    spy_dd_series = spy_cum / np.maximum.accumulate(spy_cum) - 1
    
    ax2.fill_between(dates[:len(agent_dd_series)], agent_dd_series, 0, 
                     alpha=0.5, color="blue", label="Agent DD")
    ax2.plot(dates[:len(spy_dd_series)], spy_dd_series, color="gray", 
             linewidth=1, linestyle="--", label="SPY DD")
    
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date/Period")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Benchmark comparison saved to {save_path}")
    
    return fig


def plot_ablation_results(
    results: Dict[str, Dict[str, Any]],
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Optional['plt.Figure']:
    """
    Create ablation study visualization.
    
    Args:
        results: Dictionary with ablation results
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    check_plotting_available()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    experiments = list(results.keys())
    sharpe_ratios = [results[e].get("sharpe_ratio", 0) for e in experiments]
    win_rates = [results[e].get("win_rate", 0) for e in experiments]
    
    x = np.arange(len(experiments))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, sharpe_ratios, width, 
                   label="Sharpe Ratio", color="steelblue")
    bars2 = ax.bar(x + width/2, win_rates, width, 
                   label="Win Rate", color="coral")
    
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Metric Value")
    ax.set_title("Ablation Study: Component Contributions")
    ax.set_xticks(x)
    
    # Format x-axis labels
    labels = []
    for exp in experiments:
        if exp == "full_system":
            labels.append("Full\nSystem")
        elif exp == "no_historical":
            labels.append("No\nHistorical")
        elif exp == "no_sentiment":
            labels.append("No\nSentiment")
        elif exp == "no_thompson":
            labels.append("No\nThompson")
        elif exp == "random_actions":
            labels.append("Random")
        else:
            labels.append(exp.replace("_", "\n"))
    
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Ablation results saved to {save_path}")
    
    return fig


def plot_action_distribution(
    actions: List[int],
    save_path: str = None,
    figsize: Tuple[int, int] = (8, 6)
) -> Optional['plt.Figure']:
    """Plot distribution of actions taken."""
    check_plotting_available()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    action_names = ["No Trade", "Long", "Short", "Long Vol", "Short Vol"]
    colors = ["gray", "green", "red", "blue", "orange"]
    
    counts = [actions.count(i) for i in range(5)]
    
    ax.bar(action_names, counts, color=colors)
    ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    ax.set_title("Action Distribution")
    
    # Add percentage labels
    total = sum(counts)
    for i, (count, name) in enumerate(zip(counts, action_names)):
        pct = count / total * 100 if total > 0 else 0
        ax.annotate(f'{pct:.1f}%', xy=(i, count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_returns_histogram(
    returns: List[float],
    save_path: str = None,
    figsize: Tuple[int, int] = (8, 6)
) -> Optional['plt.Figure']:
    """Plot histogram of returns."""
    check_plotting_available()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(returns, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', label='Zero')
    ax.axvline(x=np.mean(returns), color='green', linestyle='-', 
               label=f'Mean: {np.mean(returns):.2%}')
    
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Trade Returns")
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def _smooth(values: List[float], window: int = 10) -> List[float]:
    """Apply simple moving average smoothing."""
    if len(values) < window:
        return values
    
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(values[start:i+1]))
    
    return smoothed


# Alias for backward compatibility
from .metrics import format_metrics_report as create_metrics_report
