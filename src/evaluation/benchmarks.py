"""
Benchmark Strategies for EETA.

Provides baseline strategies for comparison:
- Buy & Hold SPY
- Random Agent
- Always Long
- Momentum Strategy
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from ..rl.actions import Actions

logger = logging.getLogger(__name__)


class BaseBenchmark(ABC):
    """Base class for benchmark strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass
    
    @abstractmethod
    def get_action(self, state: np.ndarray, event: Dict[str, Any]) -> int:
        """
        Get action for given state.
        
        Args:
            state: State vector
            event: Event data
            
        Returns:
            Action index
        """
        pass
    
    def run_backtest(
        self,
        events: pd.DataFrame,
        position_size: float = 0.02
    ) -> Dict[str, Any]:
        """
        Run backtest on events.
        
        Args:
            events: DataFrame with earnings events
            position_size: Position size per trade
            
        Returns:
            Backtest results
        """
        returns = []
        actions_taken = []
        
        for _, event in events.iterrows():
            # Get state (simplified - just use event data directly)
            state = np.zeros(43)  # Placeholder
            
            action = self.get_action(state, event.to_dict())
            actions_taken.append(action)
            
            if action == Actions.NO_TRADE:
                continue
            
            actual_move = event.get('actual_move', 0.0)
            expected_move = event.get('mkt_expected_move', 0.05)
            
            pnl = self._calculate_pnl(action, actual_move, expected_move, position_size)
            returns.append(pnl)
        
        return {
            'name': self.name,
            'returns': returns,
            'total_return': sum(returns),
            'n_trades': len(returns),
            'actions': actions_taken
        }
    
    def _calculate_pnl(
        self,
        action: int,
        actual_move: float,
        expected_move: float,
        position_size: float
    ) -> float:
        """Calculate P&L for action."""
        if action == Actions.NO_TRADE:
            return 0.0
        elif action == Actions.LONG_STOCK:
            return actual_move * position_size
        elif action == Actions.SHORT_STOCK:
            return -actual_move * position_size
        elif action == Actions.LONG_VOL:
            return (abs(actual_move) - expected_move) * position_size
        elif action == Actions.SHORT_VOL:
            threshold = expected_move * 0.7
            if abs(actual_move) < threshold:
                return expected_move * 0.3 * position_size
            else:
                return -(abs(actual_move) - threshold) * 0.5 * position_size
        return 0.0


class RandomBenchmark(BaseBenchmark):
    """Random action selection benchmark."""
    
    def __init__(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
    
    @property
    def name(self) -> str:
        return "Random"
    
    def get_action(self, state: np.ndarray, event: Dict[str, Any]) -> int:
        return np.random.randint(5)


class AlwaysLongBenchmark(BaseBenchmark):
    """Always buy before earnings."""
    
    @property
    def name(self) -> str:
        return "Always Long"
    
    def get_action(self, state: np.ndarray, event: Dict[str, Any]) -> int:
        return Actions.LONG_STOCK


class AlwaysShortBenchmark(BaseBenchmark):
    """Always short before earnings."""
    
    @property
    def name(self) -> str:
        return "Always Short"
    
    def get_action(self, state: np.ndarray, event: Dict[str, Any]) -> int:
        return Actions.SHORT_STOCK


class LongVolBenchmark(BaseBenchmark):
    """Always go long volatility (straddle) before earnings."""
    
    @property
    def name(self) -> str:
        return "Long Volatility"
    
    def get_action(self, state: np.ndarray, event: Dict[str, Any]) -> int:
        return Actions.LONG_VOL


class MomentumBenchmark(BaseBenchmark):
    """Simple momentum strategy: long if up-trending, short otherwise."""
    
    @property
    def name(self) -> str:
        return "Momentum"
    
    def get_action(self, state: np.ndarray, event: Dict[str, Any]) -> int:
        # Use momentum from event or state
        momentum = event.get('tech_momentum', 0.0)
        
        if momentum > 0.02:
            return Actions.LONG_STOCK
        elif momentum < -0.02:
            return Actions.SHORT_STOCK
        else:
            return Actions.NO_TRADE


class BeatRateBenchmark(BaseBenchmark):
    """Trade based on historical beat rate."""
    
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
    
    @property
    def name(self) -> str:
        return "Beat Rate"
    
    def get_action(self, state: np.ndarray, event: Dict[str, Any]) -> int:
        beat_rate = event.get('hist_beat_rate', 0.5)
        
        if beat_rate > self.threshold:
            return Actions.LONG_STOCK
        elif beat_rate < (1 - self.threshold):
            return Actions.SHORT_STOCK
        else:
            return Actions.NO_TRADE


class BuyAndHoldSPY:
    """
    Buy and Hold SPY benchmark.
    
    Calculates what return you would get from simply holding SPY
    over the same period as the earnings events.
    """
    
    @property
    def name(self) -> str:
        return "Buy & Hold SPY"
    
    def calculate_returns(
        self,
        events: pd.DataFrame,
        spy_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate SPY returns for comparison.
        
        Args:
            events: DataFrame with earnings events (for date range)
            spy_data: DataFrame with SPY price data
            
        Returns:
            Benchmark results
        """
        if spy_data.empty or events.empty:
            return {'name': self.name, 'returns': [], 'total_return': 0.0}
        
        # Get date range from events
        start_date = pd.to_datetime(events['earnings_date'].min())
        end_date = pd.to_datetime(events['earnings_date'].max())
        
        # Filter SPY data
        spy_data = spy_data.copy()
        spy_data.index = pd.to_datetime(spy_data.index)
        
        mask = (spy_data.index >= start_date) & (spy_data.index <= end_date)
        spy_period = spy_data[mask]
        
        if len(spy_period) < 2:
            return {'name': self.name, 'returns': [], 'total_return': 0.0}
        
        # Calculate returns
        close_col = 'close' if 'close' in spy_period.columns else spy_period.columns[0]
        spy_returns = spy_period[close_col].pct_change().dropna().tolist()
        
        # Total return
        start_price = spy_period[close_col].iloc[0]
        end_price = spy_period[close_col].iloc[-1]
        total_return = (end_price - start_price) / start_price
        
        return {
            'name': self.name,
            'returns': spy_returns,
            'total_return': total_return,
            'start_date': start_date,
            'end_date': end_date,
            'n_days': len(spy_returns)
        }


class BenchmarkSuite:
    """Collection of benchmarks for comparison."""
    
    def __init__(self):
        self.benchmarks = [
            RandomBenchmark(seed=42),
            AlwaysLongBenchmark(),
            AlwaysShortBenchmark(),
            LongVolBenchmark(),
            MomentumBenchmark(),
            BeatRateBenchmark()
        ]
        self.spy_benchmark = BuyAndHoldSPY()
    
    def run_all(
        self,
        events: pd.DataFrame,
        spy_data: pd.DataFrame = None,
        position_size: float = 0.02
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all benchmarks.
        
        Args:
            events: Earnings events DataFrame
            spy_data: Optional SPY data for buy-and-hold
            position_size: Position size per trade
            
        Returns:
            Dictionary of results per benchmark
        """
        results = {}
        
        # Run each benchmark
        for benchmark in self.benchmarks:
            logger.info(f"Running benchmark: {benchmark.name}")
            results[benchmark.name] = benchmark.run_backtest(events, position_size)
        
        # Run SPY benchmark if data available
        if spy_data is not None:
            results[self.spy_benchmark.name] = self.spy_benchmark.calculate_returns(
                events, spy_data
            )
        
        return results
    
    def compare_to_agent(
        self,
        agent_returns: List[float],
        events: pd.DataFrame,
        spy_data: pd.DataFrame = None,
        position_size: float = 0.02
    ) -> pd.DataFrame:
        """
        Compare agent performance to all benchmarks.
        
        Args:
            agent_returns: Agent's returns
            events: Earnings events
            spy_data: Optional SPY data
            position_size: Position size
            
        Returns:
            DataFrame with comparison
        """
        from .metrics import calculate_sharpe_ratio, calculate_max_drawdown, calculate_win_rate
        
        # Run benchmarks
        benchmark_results = self.run_all(events, spy_data, position_size)
        
        # Add agent results
        benchmark_results['EETA Agent'] = {
            'name': 'EETA Agent',
            'returns': agent_returns,
            'total_return': sum(agent_returns),
            'n_trades': len(agent_returns)
        }
        
        # Calculate metrics for each
        comparison = []
        for name, result in benchmark_results.items():
            returns = result.get('returns', [])
            
            comparison.append({
                'Strategy': name,
                'Total Return': result.get('total_return', 0.0),
                'Sharpe Ratio': calculate_sharpe_ratio(returns) if returns else 0.0,
                'Max Drawdown': calculate_max_drawdown(returns) if returns else 0.0,
                'Win Rate': calculate_win_rate(returns) if returns else 0.0,
                'N Trades': len(returns)
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('Sharpe Ratio', ascending=False)
        
        return df


def run_benchmark_comparison(
    agent_returns: List[float],
    events: pd.DataFrame,
    spy_data: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Convenience function to run benchmark comparison.
    
    Args:
        agent_returns: Agent's returns
        events: Earnings events
        spy_data: Optional SPY data
        
    Returns:
        Comparison DataFrame
    """
    suite = BenchmarkSuite()
    return suite.compare_to_agent(agent_returns, events, spy_data)
