"""
Walk-Forward Validation for EETA.

Implements walk-forward validation to prevent look-ahead bias
in time series data.

CRITICAL: This methodology ensures:
1. Training only on past data
2. Fitting scalers only on training data
3. Testing on future data
"""

import logging
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..data.preprocessor import StatePreprocessor, FEATURE_NAMES

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Implements walk-forward validation for time series.
    
    CRITICAL: Prevents look-ahead bias by:
    1. Training only on past data
    2. Fitting scalers only on training data
    3. Testing on future data
    
    Validation Scheme:
    ```
    Year:     2019    2020    2021    2022    2023    2024
    Fold 1:   [═══════ TRAIN ═══════]  [TEST]
    Fold 2:   [════════ TRAIN ════════]  [TEST]
    Fold 3:   [═════════ TRAIN ═════════]  [TEST]
    ```
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        date_column: str = 'earnings_date',
        min_train_years: int = 3,
        test_years: int = 1
    ):
        """
        Initialize validator.
        
        Args:
            data: Full dataset with date column
            date_column: Name of date column
            min_train_years: Minimum years for initial training
            test_years: Years to test in each fold
        """
        self.data = data.copy()
        self.date_column = date_column
        self.min_train_years = min_train_years
        self.test_years = test_years
        
        # Ensure date column is datetime
        self.data[date_column] = pd.to_datetime(self.data[date_column])
        
        # Sort by date
        self.data = self.data.sort_values(date_column).reset_index(drop=True)
        
        # Extract unique years
        self.years = sorted(self.data[date_column].dt.year.unique())
        
        logger.info(f"Walk-forward validator initialized with years: {self.years}")
    
    def get_folds(self) -> List[Dict[str, Any]]:
        """
        Generate train/test folds.
        
        Returns:
            List of fold dictionaries with train_data, test_data, and scaler
        """
        folds = []
        
        for i in range(len(self.years) - self.min_train_years):
            train_end_year = self.years[self.min_train_years + i - 1]
            test_year = self.years[self.min_train_years + i]
            
            # Split data
            train_mask = self.data[self.date_column].dt.year <= train_end_year
            test_mask = self.data[self.date_column].dt.year == test_year
            
            train_data = self.data[train_mask].copy()
            test_data = self.data[test_mask].copy()
            
            if len(train_data) == 0 or len(test_data) == 0:
                logger.warning(f"Skipping fold {i+1}: empty train or test set")
                continue
            
            # Get feature columns
            feature_cols = [c for c in train_data.columns 
                          if c in FEATURE_NAMES or c.startswith(('hist_', 'mkt_', 'tech_', 'sent_', 'meta_'))]
            
            # Fit preprocessor ONLY on training data
            preprocessor = StatePreprocessor()
            if feature_cols:
                train_features = train_data[feature_cols].copy()
                # Rename columns to match FEATURE_NAMES if needed
                preprocessor.fit(train_features)
            
            folds.append({
                'fold_id': i + 1,
                'train_years': f"{self.years[0]}-{train_end_year}",
                'test_year': test_year,
                'train_data': train_data,
                'test_data': test_data,
                'preprocessor': preprocessor,
                'feature_cols': feature_cols,
                'train_size': len(train_data),
                'test_size': len(test_data)
            })
            
            logger.info(
                f"Fold {i+1}: Train {self.years[0]}-{train_end_year} "
                f"({len(train_data)} samples), Test {test_year} ({len(test_data)} samples)"
            )
        
        return folds
    
    def run_validation(
        self,
        agent_factory: Callable,
        train_fn: Callable,
        test_fn: Callable,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run complete walk-forward validation.
        
        Args:
            agent_factory: Function that creates a fresh agent
            train_fn: Function to train agent on a fold
            test_fn: Function to test agent on a fold
            config: Training configuration
            
        Returns:
            Dictionary with results for each fold
        """
        folds = self.get_folds()
        all_results = []
        
        for fold in folds:
            logger.info("=" * 60)
            logger.info(f"FOLD {fold['fold_id']}: Train {fold['train_years']}, "
                       f"Test {fold['test_year']}")
            logger.info("=" * 60)
            
            # Create fresh agent
            agent = agent_factory(config)
            
            # Train on training data
            train_results = train_fn(agent, fold, config)
            
            # Test on test data (no further training)
            test_results = test_fn(agent, fold)
            
            all_results.append({
                'fold_id': fold['fold_id'],
                'train_years': fold['train_years'],
                'test_year': fold['test_year'],
                'train_size': fold['train_size'],
                'test_size': fold['test_size'],
                'train_results': train_results,
                'test_results': test_results
            })
        
        return {
            'folds': all_results,
            'aggregate': self._aggregate_results(all_results)
        }
    
    def _aggregate_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across all folds."""
        if not fold_results:
            return {}
        
        # Collect test metrics
        test_returns = []
        test_sharpes = []
        test_win_rates = []
        test_trades = []
        
        for fold in fold_results:
            test = fold.get('test_results', {})
            
            if 'total_return' in test:
                test_returns.append(test['total_return'])
            if 'sharpe_ratio' in test:
                test_sharpes.append(test['sharpe_ratio'])
            if 'win_rate' in test:
                test_win_rates.append(test['win_rate'])
            if 'n_trades' in test:
                test_trades.append(test['n_trades'])
        
        return {
            'n_folds': len(fold_results),
            'mean_return': np.mean(test_returns) if test_returns else 0.0,
            'std_return': np.std(test_returns) if test_returns else 0.0,
            'mean_sharpe': np.mean(test_sharpes) if test_sharpes else 0.0,
            'mean_win_rate': np.mean(test_win_rates) if test_win_rates else 0.0,
            'total_test_trades': sum(test_trades) if test_trades else 0,
            'fold_returns': test_returns,
            'fold_sharpes': test_sharpes
        }
    


def create_time_series_split(
    data: pd.DataFrame,
    n_splits: int = 5,
    date_column: str = 'earnings_date'
) -> List[tuple]:
    """
    Create time series cross-validation splits.
    
    Alternative to WalkForwardValidator for simpler use cases.
    
    Args:
        data: DataFrame with date column
        n_splits: Number of splits
        date_column: Name of date column
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    data = data.copy()
    data[date_column] = pd.to_datetime(data[date_column])
    data = data.sort_values(date_column).reset_index(drop=True)
    
    n = len(data)
    splits = []
    
    # Each split uses more training data
    for i in range(n_splits):
        # Training set grows with each split
        train_end = int(n * (0.5 + 0.1 * i))  # 50%, 60%, 70%, 80%, 90%
        test_end = min(int(n * (0.5 + 0.1 * (i + 1))), n)
        
        train_idx = list(range(train_end))
        test_idx = list(range(train_end, test_end))
        
        if len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    
    return splits




class WalkForwardResults:
    """Container for walk-forward validation results."""
    
    def __init__(self, results: Dict[str, Any]):
        self.results = results
        self.folds = results.get('folds', [])
        self.aggregate = results.get('aggregate', {})
    
    def get_fold(self, fold_id: int) -> Optional[Dict[str, Any]]:
        """Get results for a specific fold."""
        for fold in self.folds:
            if fold['fold_id'] == fold_id:
                return fold
        return None
    
    def get_test_returns(self) -> List[float]:
        """Get test returns for all folds."""
        return self.aggregate.get('fold_returns', [])
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        agg = self.aggregate
        
        lines = [
            "Walk-Forward Validation Results",
            "=" * 40,
            f"Number of Folds: {agg.get('n_folds', 0)}",
            f"Mean Test Return: {agg.get('mean_return', 0):.2%}",
            f"Std Test Return: {agg.get('std_return', 0):.2%}",
            f"Mean Sharpe Ratio: {agg.get('mean_sharpe', 0):.2f}",
            f"Mean Win Rate: {agg.get('mean_win_rate', 0):.1%}",
            f"Total Test Trades: {agg.get('total_test_trades', 0)}",
            "",
            "Per-Fold Returns:",
        ]
        
        for i, ret in enumerate(agg.get('fold_returns', []), 1):
            lines.append(f"  Fold {i}: {ret:.2%}")
        
        return "\n".join(lines)

def create_agent_factory(config: Dict[str, Any]) -> Callable:
    """Create a factory function that produces fresh agents."""
    def factory(cfg: Dict[str, Any]):
        from ..rl.dqn import DQNAgent
        dqn_config = cfg.get('dqn', {})
        return DQNAgent(
            state_dim=dqn_config.get('state_dim', 43),
            action_dim=dqn_config.get('action_dim', 5),
            hidden_dims=dqn_config.get('hidden_dims', [128, 64]),
            learning_rate=dqn_config.get('learning_rate', 0.001),
            gamma=dqn_config.get('gamma', 0.99),
            epsilon_start=dqn_config.get('epsilon_start', 1.0),
            epsilon_end=dqn_config.get('epsilon_end', 0.05),
            epsilon_decay=dqn_config.get('epsilon_decay', 0.995),
            batch_size=dqn_config.get('batch_size', 32),
            buffer_size=dqn_config.get('replay_buffer_size', 10000)
        )
    return factory


def train_fold(agent, fold: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Train agent on a single fold."""
    from ..environment.trading_env import EarningsTradingEnv
    
    train_data = fold['train_data']
    preprocessor = fold['preprocessor']
    
    env = EarningsTradingEnv(
        data=train_data,
        preprocessor=preprocessor,
        shuffle=True
    )
    
    n_episodes = config.get('training', {}).get('episodes_per_fold', 100)
    total_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()  # Returns only state
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)  # Returns 4 values
            agent.store_experience(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        agent.decay_epsilon()
    
    return {
        'episodes': n_episodes,
        'mean_reward': np.mean(total_rewards),
        'final_epsilon': agent.epsilon
    }


def test_fold(agent, fold: Dict[str, Any]) -> Dict[str, Any]:
    """Test agent on a single fold (no training)."""
    from ..environment.trading_env import EarningsTradingEnv
    
    test_data = fold['test_data']
    preprocessor = fold['preprocessor']
    
    env = EarningsTradingEnv(
        data=test_data,
        preprocessor=preprocessor,
        shuffle=False
    )
    
    state = env.reset()  # Returns only state
    returns = []
    actions_taken = []
    done = False
    
    # Set agent to evaluation mode (no exploration)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)  # Returns 4 values
        
        if action != 0:  # Not NO_TRADE
            returns.append(info.get('pnl_pct', 0))
        actions_taken.append(action)
        state = next_state
    
    # Restore epsilon
    agent.epsilon = original_epsilon
    
    # Calculate metrics
    returns = np.array(returns) if returns else np.array([0])
    total_return = np.sum(returns)
    n_trades = len([a for a in actions_taken if a != 0])
    win_rate = np.mean(returns > 0) if len(returns) > 0 else 0
    
    # Sharpe ratio
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    return {
        'total_return': total_return,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe,
        'actions': actions_taken
    }