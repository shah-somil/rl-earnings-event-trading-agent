"""
Performance Metrics for EETA.

Provides comprehensive trading performance metrics including:
- Return metrics
- Risk metrics
- Trading statistics
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def calculate_total_return(returns: List[float]) -> float:
    """Calculate total return from list of returns."""
    if not returns:
        return 0.0
    return sum(returns)


def calculate_cumulative_return(returns: List[float]) -> float:
    """Calculate cumulative compounded return."""
    if not returns:
        return 0.0
    return np.prod([1 + r for r in returns]) - 1


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
    annualize: bool = True
) -> float:
    """
    Calculate Sharpe ratio.
    
    Sharpe = E[R - Rf] / std(R) * sqrt(252)
    
    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate (daily)
        annualize: Whether to annualize
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    returns = np.array(returns)
    excess = returns - risk_free_rate
    
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    
    sharpe = np.mean(excess) / std
    
    if annualize:
        sharpe *= np.sqrt(252)
    
    return sharpe


def calculate_sortino_ratio(
    returns: List[float],
    target_return: float = 0.0,
    annualize: bool = True
) -> float:
    """
    Calculate Sortino ratio (uses downside deviation).
    
    Args:
        returns: List of returns
        target_return: Target return for downside calculation
        annualize: Whether to annualize
        
    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    returns = np.array(returns)
    excess = returns - target_return
    
    # Downside deviation
    downside = returns[returns < target_return]
    if len(downside) == 0:
        return float('inf') if np.mean(excess) > 0 else 0.0
    
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return 0.0
    
    sortino = np.mean(excess) / downside_std
    
    if annualize:
        sortino *= np.sqrt(252)
    
    return sortino


def calculate_max_drawdown(returns: List[float]) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        returns: List of returns
        
    Returns:
        Maximum drawdown as positive decimal
    """
    if not returns:
        return 0.0
    
    # Calculate cumulative returns
    cumulative = np.cumprod([1 + r for r in returns])
    
    # Track running maximum
    running_max = np.maximum.accumulate(cumulative)
    
    # Drawdown at each point
    drawdowns = (running_max - cumulative) / running_max
    
    return float(np.max(drawdowns))


def calculate_calmar_ratio(
    returns: List[float],
    annualize: bool = True
) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).
    
    Args:
        returns: List of returns
        annualize: Whether to annualize returns
        
    Returns:
        Calmar ratio
    """
    max_dd = calculate_max_drawdown(returns)
    if max_dd == 0:
        return 0.0
    
    total_return = calculate_cumulative_return(returns)
    
    if annualize and len(returns) > 0:
        years = len(returns) / 252
        if years > 0:
            annual_return = (1 + total_return) ** (1 / years) - 1
        else:
            annual_return = total_return
    else:
        annual_return = total_return
    
    return annual_return / max_dd


def calculate_win_rate(returns: List[float]) -> float:
    """Calculate win rate (fraction of positive returns)."""
    if not returns:
        return 0.0
    return sum(1 for r in returns if r > 0) / len(returns)


def calculate_profit_factor(returns: List[float]) -> float:
    """
    Calculate profit factor (gross profit / gross loss).
    
    Args:
        returns: List of returns
        
    Returns:
        Profit factor (>1 is profitable)
    """
    if not returns:
        return 0.0
    
    gains = sum(r for r in returns if r > 0)
    losses = abs(sum(r for r in returns if r < 0))
    
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    
    return gains / losses


def calculate_avg_win_loss(returns: List[float]) -> Dict[str, float]:
    """Calculate average win and loss sizes."""
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    
    return {
        'avg_win': np.mean(wins) if wins else 0.0,
        'avg_loss': np.mean(losses) if losses else 0.0,
        'win_loss_ratio': abs(np.mean(wins) / np.mean(losses)) if losses and wins else 0.0
    }


def calculate_volatility(
    returns: List[float],
    annualize: bool = True
) -> float:
    """Calculate return volatility."""
    if len(returns) < 2:
        return 0.0
    
    vol = np.std(returns, ddof=1)
    
    if annualize:
        vol *= np.sqrt(252)
    
    return vol


def calculate_value_at_risk(
    returns: List[float],
    confidence: float = 0.95
) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: List of returns
        confidence: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        VaR as positive decimal (loss)
    """
    if not returns:
        return 0.0
    
    var = np.percentile(returns, (1 - confidence) * 100)
    return -var if var < 0 else 0.0


def calculate_expected_shortfall(
    returns: List[float],
    confidence: float = 0.95
) -> float:
    """
    Calculate Expected Shortfall (CVaR).
    
    Average loss in the worst (1-confidence)% of cases.
    
    Args:
        returns: List of returns
        confidence: Confidence level
        
    Returns:
        Expected Shortfall as positive decimal
    """
    if not returns:
        return 0.0
    
    var = np.percentile(returns, (1 - confidence) * 100)
    tail = [r for r in returns if r <= var]
    
    if not tail:
        return 0.0
    
    return -np.mean(tail)


def calculate_all_metrics(
    returns: List[float],
    benchmark_returns: List[float] = None,
    risk_free_rate: float = 0.0
) -> Dict[str, Any]:
    """
    Calculate all performance metrics.
    
    Args:
        returns: Strategy returns
        benchmark_returns: Optional benchmark returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        # Return metrics
        'total_return': calculate_total_return(returns),
        'cumulative_return': calculate_cumulative_return(returns),
        'annualized_return': 0.0,
        
        # Risk metrics
        'volatility': calculate_volatility(returns),
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate),
        'sortino_ratio': calculate_sortino_ratio(returns),
        'max_drawdown': calculate_max_drawdown(returns),
        'calmar_ratio': calculate_calmar_ratio(returns),
        'var_95': calculate_value_at_risk(returns, 0.95),
        'cvar_95': calculate_expected_shortfall(returns, 0.95),
        
        # Trading statistics
        'win_rate': calculate_win_rate(returns),
        'profit_factor': calculate_profit_factor(returns),
        'n_trades': len(returns),
    }
    
    # Win/loss analysis
    wl = calculate_avg_win_loss(returns)
    metrics.update(wl)
    
    # Annualized return
    if len(returns) > 0:
        years = len(returns) / 252
        if years > 0:
            cum_ret = calculate_cumulative_return(returns)
            metrics['annualized_return'] = (1 + cum_ret) ** (1 / years) - 1
    
    # Benchmark comparison
    if benchmark_returns and len(benchmark_returns) == len(returns):
        metrics['benchmark_return'] = calculate_cumulative_return(benchmark_returns)
        metrics['excess_return'] = metrics['cumulative_return'] - metrics['benchmark_return']
        metrics['benchmark_sharpe'] = calculate_sharpe_ratio(benchmark_returns, risk_free_rate)
        
        # Information ratio
        active_returns = [r - b for r, b in zip(returns, benchmark_returns)]
        if np.std(active_returns) > 0:
            metrics['information_ratio'] = np.mean(active_returns) / np.std(active_returns) * np.sqrt(252)
        else:
            metrics['information_ratio'] = 0.0
    
    return metrics


def format_metrics_report(metrics: Dict[str, Any]) -> str:
    """Format metrics as human-readable report."""
    lines = [
        "=" * 50,
        "PERFORMANCE METRICS REPORT",
        "=" * 50,
        "",
        "Return Metrics:",
        f"  Total Return: {metrics.get('total_return', 0):.2%}",
        f"  Cumulative Return: {metrics.get('cumulative_return', 0):.2%}",
        f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}",
        "",
        "Risk Metrics:",
        f"  Volatility: {metrics.get('volatility', 0):.2%}",
        f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
        f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}",
        f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}",
        f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}",
        f"  VaR (95%): {metrics.get('var_95', 0):.2%}",
        f"  CVaR (95%): {metrics.get('cvar_95', 0):.2%}",
        "",
        "Trading Statistics:",
        f"  Number of Trades: {metrics.get('n_trades', 0)}",
        f"  Win Rate: {metrics.get('win_rate', 0):.1%}",
        f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}",
        f"  Avg Win: {metrics.get('avg_win', 0):.2%}",
        f"  Avg Loss: {metrics.get('avg_loss', 0):.2%}",
        f"  Win/Loss Ratio: {metrics.get('win_loss_ratio', 0):.2f}",
    ]
    
    if 'excess_return' in metrics:
        lines.extend([
            "",
            "Benchmark Comparison:",
            f"  Benchmark Return: {metrics.get('benchmark_return', 0):.2%}",
            f"  Excess Return: {metrics.get('excess_return', 0):.2%}",
            f"  Information Ratio: {metrics.get('information_ratio', 0):.2f}",
        ])
    
    lines.append("=" * 50)
    
    return "\n".join(lines)
