"""
Backtesting Engine for EETA.

Provides comprehensive backtesting capabilities:
- Historical simulation
- Performance tracking
- Transaction costs
- Risk-adjusted metrics
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd

from ..rl.actions import Actions
from ..environment.action_simulator import ActionSimulator
from .metrics import calculate_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Single trade record."""
    date: datetime
    ticker: str
    action: int
    action_name: str
    position_size: float
    entry_price: float
    actual_move: float
    expected_move: float
    pnl_gross: float
    pnl_net: float
    costs: float
    cumulative_pnl: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest results."""
    trades: List[BacktestTrade]
    metrics: Dict[str, Any]
    equity_curve: List[float]
    drawdown_curve: List[float]
    
    @property
    def total_return(self) -> float:
        return self.metrics.get('total_return', 0.0)
    
    @property
    def sharpe_ratio(self) -> float:
        return self.metrics.get('sharpe_ratio', 0.0)
    
    @property
    def max_drawdown(self) -> float:
        return self.metrics.get('max_drawdown', 0.0)
    
    @property
    def win_rate(self) -> float:
        return self.metrics.get('win_rate', 0.0)
    
    @property
    def n_trades(self) -> int:
        return len([t for t in self.trades if t.action != Actions.NO_TRADE])
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        return pd.DataFrame([
            {
                'date': t.date,
                'ticker': t.ticker,
                'action': t.action_name,
                'position_size': t.position_size,
                'actual_move': t.actual_move,
                'pnl_net': t.pnl_net,
                'cumulative_pnl': t.cumulative_pnl
            }
            for t in self.trades
        ])


class Backtester:
    """
    Backtesting engine for evaluating trading strategies.
    
    Features:
    - Transaction cost modeling
    - Slippage simulation
    - Risk limit enforcement
    - Detailed trade logging
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        max_position_size: float = 0.05
    ):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Cost per trade as fraction
            slippage: Slippage as fraction
            max_position_size: Maximum position size
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position_size = max_position_size
        
        self.simulator = ActionSimulator(
            transaction_cost=transaction_cost,
            slippage=slippage
        )
    
    def run(
        self,
        events: pd.DataFrame,
        action_fn,
        position_size_fn = None
    ) -> BacktestResult:
        """
        Run backtest on historical events.
        
        Args:
            events: DataFrame with earnings events
            action_fn: Function(state, event) -> action
            position_size_fn: Optional function(state, event, action) -> size
            
        Returns:
            BacktestResult with trades and metrics
        """
        trades = []
        equity_curve = [self.initial_capital]
        cumulative_pnl = 0.0
        
        for idx, event in events.iterrows():
            # Get state (simplified)
            state = self._event_to_state(event)
            
            # Get action
            action = action_fn(state, event.to_dict())
            
            # Get position size
            if position_size_fn:
                position_size = position_size_fn(state, event.to_dict(), action)
            else:
                position_size = 0.02  # Default 2%
            
            # Enforce max position size
            position_size = min(position_size, self.max_position_size)
            
            # Skip no-trade
            if action == Actions.NO_TRADE:
                continue
            
            # Get event data
            actual_move = event.get('actual_move', 0.0)
            expected_move = event.get('mkt_expected_move', 0.05)
            
            # Simulate P&L
            result = self.simulator.simulate_pnl(
                action=action,
                actual_move=actual_move,
                expected_move=expected_move,
                position_size=position_size
            )
            
            pnl_net = result['net_pnl_pct']
            cumulative_pnl += pnl_net
            
            # Record trade
            trade = BacktestTrade(
                date=event.get('earnings_date', datetime.now()),
                ticker=event.get('ticker', 'UNKNOWN'),
                action=action,
                action_name=Actions.name(action),
                position_size=position_size,
                entry_price=event.get('pre_close', 100.0),
                actual_move=actual_move,
                expected_move=expected_move,
                pnl_gross=result['gross_pnl_pct'],
                pnl_net=pnl_net,
                costs=result['costs'],
                cumulative_pnl=cumulative_pnl
            )
            trades.append(trade)
            
            # Update equity curve
            current_equity = equity_curve[-1] * (1 + pnl_net)
            equity_curve.append(current_equity)
        
        # Calculate metrics
        returns = [t.pnl_net for t in trades]
        metrics = calculate_all_metrics(returns)
        
        # Calculate drawdown curve
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown_curve = (running_max - equity_array) / running_max
        
        return BacktestResult(
            trades=trades,
            metrics=metrics,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve.tolist()
        )
    
    def _event_to_state(self, event: pd.Series) -> np.ndarray:
        """Convert event to state vector (simplified)."""
        from ..data.preprocessor import FEATURE_NAMES, STATE_DIM
        
        state = np.zeros(STATE_DIM)
        
        for i, name in enumerate(FEATURE_NAMES):
            if name in event.index:
                value = event[name]
                if pd.notna(value):
                    state[i] = value
        
        return state
    
    def run_with_agent(
        self,
        events: pd.DataFrame,
        agent,
        thompson_sampler = None,
        use_greedy: bool = True
    ) -> BacktestResult:
        """
        Run backtest with a trained agent.
        
        Args:
            events: DataFrame with earnings events
            agent: Trained DQNAgent
            thompson_sampler: Optional Thompson Sampler for sizing
            use_greedy: Whether to use greedy action selection
            
        Returns:
            BacktestResult
        """
        def action_fn(state, event):
            return agent.select_action(state, greedy=use_greedy)
        
        def size_fn(state, event, action):
            if thompson_sampler:
                confidence = state[35] if len(state) > 35 else 0.5
                volatility = state[20] if len(state) > 20 else 0.5
                _, size = thompson_sampler.select_size(
                    confidence=confidence,
                    volatility=volatility,
                    greedy=use_greedy
                )
                return size
            return 0.02
        
        return self.run(events, action_fn, size_fn)


class WalkForwardBacktester:
    """
    Walk-forward backtesting with periodic retraining.
    
    Simulates realistic deployment where model is retrained
    periodically on new data.
    """
    
    def __init__(
        self,
        train_period_months: int = 12,
        test_period_months: int = 3,
        **backtester_kwargs
    ):
        """
        Initialize walk-forward backtester.
        
        Args:
            train_period_months: Months of data for training
            test_period_months: Months of data for testing
            **backtester_kwargs: Arguments for Backtester
        """
        self.train_period_months = train_period_months
        self.test_period_months = test_period_months
        self.backtester = Backtester(**backtester_kwargs)
    
    def run(
        self,
        events: pd.DataFrame,
        agent_factory,
        train_fn,
        date_column: str = 'earnings_date'
    ) -> Dict[str, Any]:
        """
        Run walk-forward backtest.
        
        Args:
            events: Full dataset
            agent_factory: Function to create new agent
            train_fn: Function to train agent
            date_column: Name of date column
            
        Returns:
            Combined results from all periods
        """
        events = events.copy()
        events[date_column] = pd.to_datetime(events[date_column])
        events = events.sort_values(date_column)
        
        all_results = []
        all_trades = []
        
        # Generate periods
        min_date = events[date_column].min()
        max_date = events[date_column].max()
        
        current_start = min_date
        
        while current_start < max_date:
            train_end = current_start + pd.DateOffset(months=self.train_period_months)
            test_end = train_end + pd.DateOffset(months=self.test_period_months)
            
            # Get train and test data
            train_mask = (events[date_column] >= current_start) & (events[date_column] < train_end)
            test_mask = (events[date_column] >= train_end) & (events[date_column] < test_end)
            
            train_data = events[train_mask]
            test_data = events[test_mask]
            
            if len(train_data) < 50 or len(test_data) < 10:
                current_start = train_end
                continue
            
            logger.info(f"Period: Train {current_start.date()} to {train_end.date()}, "
                       f"Test {train_end.date()} to {test_end.date()}")
            
            # Train agent
            agent = agent_factory()
            train_fn(agent, train_data)
            
            # Test
            result = self.backtester.run_with_agent(test_data, agent)
            
            all_results.append({
                'train_start': current_start,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end,
                'result': result
            })
            all_trades.extend(result.trades)
            
            # Move to next period
            current_start = train_end
        
        # Aggregate metrics
        all_returns = [t.pnl_net for t in all_trades]
        combined_metrics = calculate_all_metrics(all_returns)
        
        return {
            'periods': all_results,
            'all_trades': all_trades,
            'combined_metrics': combined_metrics,
            'n_periods': len(all_results),
            'total_trades': len(all_trades)
        }


def quick_backtest(
    events: pd.DataFrame,
    strategy: str = 'always_long'
) -> BacktestResult:
    """
    Quick backtest with predefined strategy.
    
    Args:
        events: Earnings events
        strategy: 'always_long', 'always_short', 'momentum', 'random'
        
    Returns:
        BacktestResult
    """
    from .benchmarks import (
        AlwaysLongBenchmark, 
        AlwaysShortBenchmark,
        MomentumBenchmark,
        RandomBenchmark
    )
    
    strategies = {
        'always_long': AlwaysLongBenchmark(),
        'always_short': AlwaysShortBenchmark(),
        'momentum': MomentumBenchmark(),
        'random': RandomBenchmark()
    }
    
    benchmark = strategies.get(strategy, strategies['random'])
    
    backtester = Backtester()
    return backtester.run(
        events,
        action_fn=lambda state, event: benchmark.get_action(state, event)
    )
