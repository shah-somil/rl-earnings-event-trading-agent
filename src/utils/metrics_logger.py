"""
EETA Training Metrics Logger

Saves comprehensive metrics during training for visualization:
- DQN: Loss, Q-values, epsilon, rewards
- Thompson Sampling: Beta parameters, selection counts, win rates per bucket
- Episode: Actions, returns, win rates
- Curriculum: Difficulty progression
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode: int
    total_reward: float
    total_pnl: float
    n_trades: int
    win_rate: float
    sharpe: float
    
    # Action distribution
    action_counts: Dict[int, int] = field(default_factory=dict)
    
    # DQN metrics
    avg_loss: float = 0.0
    avg_q_value: float = 0.0
    max_q_value: float = 0.0
    min_q_value: float = 0.0
    epsilon: float = 1.0
    
    # Thompson Sampling metrics
    thompson_selections: Dict[int, int] = field(default_factory=dict)
    thompson_alphas: List[float] = field(default_factory=list)
    thompson_betas: List[float] = field(default_factory=list)
    thompson_win_rates: List[float] = field(default_factory=list)
    
    # Curriculum
    difficulty: str = "easy"
    
    # Timing
    duration_seconds: float = 0.0


class MetricsLogger:
    """
    Comprehensive metrics logger for EETA training.
    
    Captures:
    1. DQN Learning: Loss curve, Q-value distribution, epsilon decay
    2. Thompson Sampling: Beta evolution, bucket selection, win rates
    3. Training: Rewards, P&L, actions, win rates over episodes
    4. Curriculum: Difficulty progression
    """
    
    def __init__(self, experiment_name: str = None, save_dir: str = "experiments/metrics"):
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(save_dir) / self.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Episode-level metrics
        self.episodes: List[EpisodeMetrics] = []
        
        # Step-level metrics (for detailed plots)
        self.step_losses: List[float] = []
        self.step_q_values: List[float] = []
        self.step_rewards: List[float] = []
        self.step_actions: List[int] = []
        
        # Thompson Sampling history
        self.thompson_history: List[Dict] = []
        
        # Current episode tracking
        self.current_episode = 0
        self.episode_losses: List[float] = []
        self.episode_q_values: List[float] = []
        self.episode_rewards: List[float] = []
        self.episode_actions: List[int] = []
        self.episode_pnls: List[float] = []
        self.episode_start_time: float = None
        
        print(f"📊 Metrics logger initialized: {self.save_dir}")
    
    def start_episode(self, episode: int):
        """Start tracking a new episode."""
        import time
        self.current_episode = episode
        self.episode_losses = []
        self.episode_q_values = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_pnls = []
        self.episode_start_time = time.time()
    
    def log_step(
        self,
        action: int,
        reward: float,
        pnl: float = 0.0,
        loss: float = None,
        q_values: np.ndarray = None
    ):
        """Log metrics for a single step."""
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_pnls.append(pnl)
        self.step_actions.append(action)
        self.step_rewards.append(reward)
        
        if loss is not None:
            self.episode_losses.append(loss)
            self.step_losses.append(loss)
        
        if q_values is not None:
            avg_q = float(np.mean(q_values))
            self.episode_q_values.append(avg_q)
            self.step_q_values.append(avg_q)
    
    def log_dqn_training(self, loss: float, q_values: np.ndarray = None):
        """Log DQN training step."""
        self.episode_losses.append(loss)
        self.step_losses.append(loss)
        
        if q_values is not None:
            self.episode_q_values.append(float(np.mean(q_values)))
    
    def log_thompson_update(
        self,
        bucket_idx: int,
        position_size: float,
        alphas: List[float],
        betas: List[float],
        profit: bool
    ):
        """Log Thompson Sampling update."""
        self.thompson_history.append({
            'episode': self.current_episode,
            'bucket_idx': bucket_idx,
            'position_size': position_size,
            'alphas': alphas.copy() if isinstance(alphas, list) else list(alphas),
            'betas': betas.copy() if isinstance(betas, list) else list(betas),
            'profit': profit
        })
    
    def end_episode(
        self,
        epsilon: float,
        difficulty: str = "easy",
        thompson_sampler=None
    ):
        """End episode and compute summary metrics."""
        import time
        
        duration = time.time() - self.episode_start_time if self.episode_start_time else 0
        
        # Compute action distribution
        action_counts = {}
        for a in range(5):
            action_counts[a] = self.episode_actions.count(a)
        
        # Compute metrics
        total_reward = sum(self.episode_rewards)
        total_pnl = sum(self.episode_pnls)
        n_trades = len([a for a in self.episode_actions if a != 0])
        
        # Win rate
        wins = sum(1 for p in self.episode_pnls if p > 0)
        win_rate = wins / max(len(self.episode_pnls), 1)
        
        # Sharpe
        if len(self.episode_pnls) > 1 and np.std(self.episode_pnls) > 0:
            sharpe = np.mean(self.episode_pnls) / np.std(self.episode_pnls) * np.sqrt(len(self.episode_pnls))
        else:
            sharpe = 0.0
        
        # DQN metrics
        avg_loss = np.mean(self.episode_losses) if self.episode_losses else 0.0
        avg_q = np.mean(self.episode_q_values) if self.episode_q_values else 0.0
        max_q = max(self.episode_q_values) if self.episode_q_values else 0.0
        min_q = min(self.episode_q_values) if self.episode_q_values else 0.0
        
        # Thompson Sampling metrics
        thompson_selections = {}
        thompson_alphas = []
        thompson_betas = []
        thompson_win_rates = []
        
        if thompson_sampler is not None:
            thompson_alphas = list(thompson_sampler.alphas)
            thompson_betas = list(thompson_sampler.betas)
            for i in range(len(thompson_sampler.alphas)):
                alpha = thompson_sampler.alphas[i]
                beta = thompson_sampler.betas[i]
                thompson_win_rates.append(alpha / (alpha + beta))
            thompson_selections = dict(thompson_sampler.selection_counts) if hasattr(thompson_sampler, 'selection_counts') else {}
        
        # Create episode metrics
        metrics = EpisodeMetrics(
            episode=self.current_episode,
            total_reward=total_reward,
            total_pnl=total_pnl,
            n_trades=n_trades,
            win_rate=win_rate,
            sharpe=sharpe,
            action_counts=action_counts,
            avg_loss=avg_loss,
            avg_q_value=avg_q,
            max_q_value=max_q,
            min_q_value=min_q,
            epsilon=epsilon,
            thompson_selections=thompson_selections,
            thompson_alphas=thompson_alphas,
            thompson_betas=thompson_betas,
            thompson_win_rates=thompson_win_rates,
            difficulty=difficulty,
            duration_seconds=duration
        )
        
        self.episodes.append(metrics)
        
        return metrics
    
    def save(self):
        """Save all metrics to files."""
        # Save episode metrics as JSON
        episodes_data = [asdict(ep) for ep in self.episodes]
        with open(self.save_dir / "episodes.json", 'w') as f:
            json.dump(episodes_data, f, indent=2, default=str)
        
        # Save episode metrics as CSV for easy plotting
        episodes_df = pd.DataFrame([{
            'episode': ep.episode,
            'total_reward': ep.total_reward,
            'total_pnl': ep.total_pnl,
            'n_trades': ep.n_trades,
            'win_rate': ep.win_rate,
            'sharpe': ep.sharpe,
            'avg_loss': ep.avg_loss,
            'avg_q_value': ep.avg_q_value,
            'epsilon': ep.epsilon,
            'difficulty': ep.difficulty,
            'action_0': ep.action_counts.get(0, 0),
            'action_1': ep.action_counts.get(1, 0),
            'action_2': ep.action_counts.get(2, 0),
            'action_3': ep.action_counts.get(3, 0),
            'action_4': ep.action_counts.get(4, 0),
        } for ep in self.episodes])
        episodes_df.to_csv(self.save_dir / "episodes.csv", index=False)
        
        # Save step-level data
        steps_df = pd.DataFrame({
            'loss': self.step_losses[:len(self.step_q_values)] if self.step_losses else [],
            'q_value': self.step_q_values[:len(self.step_losses)] if self.step_q_values else [],
        })
        if not steps_df.empty:
            steps_df.to_csv(self.save_dir / "steps.csv", index=False)
        
        # Save Thompson Sampling history
        if self.thompson_history:
            with open(self.save_dir / "thompson_history.json", 'w') as f:
                json.dump(self.thompson_history, f, indent=2)
        
        print(f"💾 Metrics saved to {self.save_dir}")
        
        return self.save_dir
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all training."""
        if not self.episodes:
            return {}
        
        final = self.episodes[-1]
        best_sharpe_ep = max(self.episodes, key=lambda x: x.sharpe)
        
        return {
            'total_episodes': len(self.episodes),
            'final_epsilon': final.epsilon,
            'final_sharpe': final.sharpe,
            'final_win_rate': final.win_rate,
            'final_pnl': final.total_pnl,
            'best_sharpe': best_sharpe_ep.sharpe,
            'best_sharpe_episode': best_sharpe_ep.episode,
            'avg_loss_final': final.avg_loss,
            'metrics_path': str(self.save_dir)
        }


def load_metrics(metrics_dir: str) -> Dict[str, Any]:
    """Load saved metrics from directory."""
    metrics_dir = Path(metrics_dir)
    
    result = {}
    
    # Load episodes
    episodes_path = metrics_dir / "episodes.csv"
    if episodes_path.exists():
        result['episodes'] = pd.read_csv(episodes_path)
    
    # Load episodes JSON (has more detail)
    episodes_json_path = metrics_dir / "episodes.json"
    if episodes_json_path.exists():
        with open(episodes_json_path) as f:
            result['episodes_detail'] = json.load(f)
    
    # Load steps
    steps_path = metrics_dir / "steps.csv"
    if steps_path.exists():
        result['steps'] = pd.read_csv(steps_path)
    
    # Load Thompson history
    thompson_path = metrics_dir / "thompson_history.json"
    if thompson_path.exists():
        with open(thompson_path) as f:
            result['thompson_history'] = json.load(f)
    
    return result