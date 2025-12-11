"""
Trading Environment for EETA.

Implements a Gym-style environment for training the DQN agent
on earnings events.
"""

import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ..rl.actions import Actions
from ..data.preprocessor import StatePreprocessor, FEATURE_NAMES, STATE_DIM
from .action_simulator import ActionSimulator, RewardCalculator

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Result of a single trade."""
    ticker: str
    date: Any
    action: int
    action_name: str
    position_size: float
    actual_move: float
    expected_move: float
    pnl_pct: float
    reward: float
    reward_components: Dict[str, float]


class EarningsTradingEnv:
    """
    Trading environment for earnings events.
    
    Presents one earnings event at a time, agent selects action,
    environment returns reward based on actual outcome.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        preprocessor: StatePreprocessor = None,
        action_simulator: ActionSimulator = None,
        reward_calculator: RewardCalculator = None,
        shuffle: bool = True,
        seed: int = None
    ):
        """
        Initialize environment.
        
        Args:
            data: DataFrame with earnings events
            preprocessor: Feature preprocessor
            action_simulator: Action P&L simulator
            reward_calculator: Reward calculator
            shuffle: Whether to shuffle events each episode
            seed: Random seed
        """
        self.data = data.copy()
        self.shuffle = shuffle
        
        # Set seed
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize components
        self.preprocessor = preprocessor or StatePreprocessor()
        self.action_simulator = action_simulator or ActionSimulator()
        self.reward_calculator = reward_calculator or RewardCalculator()
        
        # Fit preprocessor if not already fitted
        if not self.preprocessor.fitted:
            logger.info("Fitting preprocessor on environment data")
            feature_cols = [c for c in self.data.columns if c in FEATURE_NAMES]
            if feature_cols:
                self.preprocessor.fit(self.data[feature_cols])
        
        # State
        self.current_idx = 0
        self.event_order = np.arange(len(self.data))
        self.done = False
        
        # Tracking
        self.episode_trades = []
        self.episode_pnl = 0.0
        
    @property
    def observation_space_dim(self) -> int:
        """Get observation space dimension."""
        return STATE_DIM
    
    @property
    def action_space_dim(self) -> int:
        """Get action space dimension."""
        return 5
    
    @property
    def num_events(self) -> int:
        """Get number of events in dataset."""
        return len(self.data)
    
    def reset(self) -> np.ndarray:
        """
        Reset environment for new episode.
        
        Returns:
            Initial state observation
        """
        self.current_idx = 0
        self.done = False
        self.episode_trades = []
        self.episode_pnl = 0.0
        
        # Shuffle if enabled
        if self.shuffle:
            np.random.shuffle(self.event_order)
        
        # Return first state
        return self._get_current_state()
    
    def step(
        self,
        action: int,
        position_size: float = 0.02
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take action and return result.
        
        Args:
            action: Action index (0-4)
            position_size: Position size as fraction
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.done:
            logger.warning("Step called on done environment")
            return self._get_current_state(), 0.0, True, {}
        
        # Get current event
        event = self._get_current_event()
        
        # Get actual move and expected move
        actual_move = event.get('actual_move', 0.0)
        expected_move = event.get('mkt_expected_move', 0.05)
        confidence = event.get('meta_overall_confidence', 0.5)
        
        # Simulate P&L
        pnl_result = self.action_simulator.simulate_pnl(
            action=action,
            actual_move=actual_move,
            expected_move=expected_move,
            position_size=position_size
        )
        
        pnl_pct = pnl_result['net_pnl_pct']
        
        # Calculate if NO_TRADE would have avoided a loss
        would_have_lost = False
        if action == Actions.NO_TRADE:
            # Check what the best alternative action would have earned
            _, best_alt_pnl = self.reward_calculator.calculate_counterfactual(
                actual_move, expected_move, confidence
            )
            would_have_lost = best_alt_pnl < 0
        
        # Calculate reward
        reward, reward_components = self.reward_calculator.calculate(
            action=action,
            pnl_pct=pnl_pct,
            confidence=confidence,
            position_size=position_size,
            would_have_lost=would_have_lost,
            actual_move=actual_move
        )
        
        # Track trade
        trade_result = TradeResult(
            ticker=event.get('ticker', 'UNKNOWN'),
            date=event.get('earnings_date', None),
            action=action,
            action_name=Actions.name(action),
            position_size=position_size,
            actual_move=actual_move,
            expected_move=expected_move,
            pnl_pct=pnl_pct,
            reward=reward,
            reward_components=reward_components
        )
        self.episode_trades.append(trade_result)
        self.episode_pnl += pnl_pct
        
        # Move to next event
        self.current_idx += 1
        
        # Check if done
        if self.current_idx >= len(self.data):
            self.done = True
        
        # Get next state
        next_state = self._get_current_state() if not self.done else np.zeros(STATE_DIM)
        
        # Info dict
        info = {
            'ticker': trade_result.ticker,
            'date': trade_result.date,
            'action_name': trade_result.action_name,
            'actual_move': actual_move,
            'pnl_pct': pnl_pct,
            'episode_pnl': self.episode_pnl,
            'events_remaining': len(self.data) - self.current_idx,
        }
        
        return next_state, reward, self.done, info
    
    def _get_current_event(self) -> Dict[str, Any]:
        """Get current event as dictionary."""
        idx = self.event_order[self.current_idx]
        return self.data.iloc[idx].to_dict()
    
    def _get_current_state(self) -> np.ndarray:
        """Get current state as normalized vector."""
        if self.done or self.current_idx >= len(self.data):
            return np.zeros(STATE_DIM)
        
        event = self._get_current_event()
        
        # Extract features
        features = {name: event.get(name, 0.0) for name in FEATURE_NAMES}
        
        # Transform to state
        if self.preprocessor.fitted:
            state = self.preprocessor.transform(features)
        else:
            state = np.array([features.get(name, 0.0) for name in FEATURE_NAMES])
        
        return state
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics for current episode."""
        if not self.episode_trades:
            return {}
        
        trades = [t for t in self.episode_trades if t.action != Actions.NO_TRADE]
        no_trades = [t for t in self.episode_trades if t.action == Actions.NO_TRADE]
        
        pnls = [t.pnl_pct for t in trades]
        rewards = [t.reward for t in self.episode_trades]
        
        return {
            'total_events': len(self.episode_trades),
            'trades_taken': len(trades),
            'no_trade_count': len(no_trades),
            'total_pnl': self.episode_pnl,
            'avg_pnl': np.mean(pnls) if pnls else 0.0,
            'win_rate': np.mean([p > 0 for p in pnls]) if pnls else 0.0,
            'total_reward': sum(rewards),
            'avg_reward': np.mean(rewards) if rewards else 0.0,
            'action_distribution': self._get_action_distribution(),
        }
    
    def _get_action_distribution(self) -> Dict[str, int]:
        """Get distribution of actions taken."""
        counts = {Actions.name(i): 0 for i in range(5)}
        for trade in self.episode_trades:
            counts[trade.action_name] += 1
        return counts


class BatchedTradingEnv:
    """
    Environment that processes multiple events as a single "episode".
    
    Useful for training where we want to batch multiple earnings events
    together.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        batch_size: int = 50,
        **kwargs
    ):
        """
        Initialize batched environment.
        
        Args:
            data: DataFrame with earnings events
            batch_size: Number of events per episode
            **kwargs: Additional arguments for base environment
        """
        self.full_data = data.copy()
        self.batch_size = batch_size
        self.kwargs = kwargs
        
        self.current_batch_idx = 0
        self.env = None
        
        self._create_batches()
    
    def _create_batches(self):
        """Create batch indices."""
        n = len(self.full_data)
        self.batch_indices = []
        
        for i in range(0, n, self.batch_size):
            end = min(i + self.batch_size, n)
            self.batch_indices.append((i, end))
    
    @property
    def num_batches(self) -> int:
        """Get number of batches."""
        return len(self.batch_indices)
    
    def reset(self, batch_idx: int = None) -> np.ndarray:
        """
        Reset to a specific batch or next batch.
        
        Args:
            batch_idx: Specific batch index, or None for next batch
            
        Returns:
            Initial state
        """
        if batch_idx is not None:
            self.current_batch_idx = batch_idx
        else:
            self.current_batch_idx = (self.current_batch_idx + 1) % self.num_batches
        
        start, end = self.batch_indices[self.current_batch_idx]
        batch_data = self.full_data.iloc[start:end].copy()
        
        self.env = EarningsTradingEnv(batch_data, **self.kwargs)
        return self.env.reset()
    
    def step(self, action: int, position_size: float = 0.02):
        """Take step in current batch."""
        if self.env is None:
            raise ValueError("Must call reset() first")
        return self.env.step(action, position_size)
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get stats for current batch."""
        if self.env is None:
            return {}
        return self.env.get_episode_stats()


def create_environment_from_fold(
    fold_data: pd.DataFrame,
    preprocessor: StatePreprocessor,
    **kwargs
) -> EarningsTradingEnv:
    """
    Create environment from a walk-forward fold.
    
    Args:
        fold_data: DataFrame for this fold
        preprocessor: Pre-fitted preprocessor
        **kwargs: Additional environment arguments
        
    Returns:
        Configured environment
    """
    return EarningsTradingEnv(
        data=fold_data,
        preprocessor=preprocessor,
        shuffle=kwargs.get('shuffle', True),
        seed=kwargs.get('seed', None)
    )
