"""
EETA - Earnings Event Trading Agent

A Multi-Agent Reinforcement Learning System for Intelligent Earnings-Based Trading.

Components:
- agents: Specialized analysis agents (Historical, Sentiment, Market)
- rl: Reinforcement learning (DQN, Thompson Sampling)
- environment: Trading environment and simulation
- data: Data fetching, preprocessing, and dataset building
- training: Training infrastructure with curriculum learning
- evaluation: Backtesting, metrics, and ablation studies
- risk: Risk management and controls
- utils: Configuration and logging utilities
"""

__version__ = "2.0.0"
__author__ = "EETA Project"

# Lazy imports to avoid import errors when dependencies are missing
def __getattr__(name):
    """Lazy import of modules."""
    if name == 'DQNAgent':
        from .rl.dqn import DQNAgent
        return DQNAgent
    elif name == 'Actions':
        from .rl.dqn import Actions
        return Actions
    elif name == 'ThompsonSampler':
        from .rl.thompson import ThompsonSampler
        return ThompsonSampler
    elif name == 'CostAwareOrchestrator':
        from .agents.orchestrator import CostAwareOrchestrator
        return CostAwareOrchestrator
    elif name == 'EarningsTradingEnv':
        from .environment.trading_env import EarningsTradingEnv
        return EarningsTradingEnv
    elif name == 'build_dataset':
        from .data.dataset_builder import build_dataset
        return build_dataset
    elif name == 'EETATrainer':
        from .training.train import EETATrainer
        return EETATrainer
    elif name == 'RiskController':
        from .risk.controller import RiskController
        return RiskController
    elif name == 'get_config':
        from .utils.config import get_config
        return get_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'DQNAgent',
    'Actions',
    'ThompsonSampler',
    'CostAwareOrchestrator',
    'EarningsTradingEnv',
    'build_dataset',
    'EETATrainer',
    'RiskController',
    'get_config',
]