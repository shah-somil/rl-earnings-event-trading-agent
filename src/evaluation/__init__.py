"""
EETA Evaluation Package.

Provides backtesting, metrics, benchmarks, and ablation studies.
"""

from .metrics import (
    calculate_total_return,
    calculate_cumulative_return,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_volatility,
    calculate_value_at_risk,
    calculate_expected_shortfall,
    calculate_all_metrics,
    format_metrics_report
)
from .benchmarks import (
    BaseBenchmark,
    RandomBenchmark,
    AlwaysLongBenchmark,
    MomentumBenchmark,
    BeatRateBenchmark,
    BuyAndHoldSPY,
    BenchmarkSuite,
    run_benchmark_comparison
)
from .ablation import (
    AblationExperiment,
    AblationStudy,
    run_ablation_study
)
from .backtester import (
    Backtester,
    BacktestResult,
    BacktestTrade,
    WalkForwardBacktester,
    quick_backtest
)

__all__ = [
    # Metrics
    'calculate_total_return',
    'calculate_cumulative_return',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_max_drawdown',
    'calculate_calmar_ratio',
    'calculate_win_rate',
    'calculate_profit_factor',
    'calculate_volatility',
    'calculate_value_at_risk',
    'calculate_expected_shortfall',
    'calculate_all_metrics',
    'format_metrics_report',
    # Benchmarks
    'BaseBenchmark',
    'RandomBenchmark',
    'AlwaysLongBenchmark',
    'MomentumBenchmark',
    'BeatRateBenchmark',
    'BuyAndHoldSPY',
    'BenchmarkSuite',
    'run_benchmark_comparison',
    # Ablation
    'AblationExperiment',
    'AblationStudy',
    'run_ablation_study',
    # Backtester
    'Backtester',
    'BacktestResult',
    'BacktestTrade',
    'WalkForwardBacktester',
    'quick_backtest',
]