"""
Configuration management for EETA.

Loads YAML configuration files and provides easy access to parameters.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. Defaults to configs/default.yaml
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Find project root
        current = Path(__file__).parent.parent.parent
        config_path = current / "configs" / "default.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


@dataclass
class DQNConfig:
    """DQN-specific configuration."""
    state_dim: int = 43
    action_dim: int = 5
    hidden_dims: list = field(default_factory=lambda: [128, 64])
    dropout: float = 0.2
    learning_rate: float = 0.001
    gamma: float = 0.99
    tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    batch_size: int = 32
    replay_buffer_size: int = 10000
    min_replay_size: int = 1000
    target_update_freq: int = 100
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DQNConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ThompsonConfig:
    """Thompson Sampling configuration."""
    num_buckets: int = 5
    size_buckets: list = field(default_factory=lambda: [0.005, 0.01, 0.02, 0.03, 0.05])
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    confidence_weight: float = 2.0
    volatility_penalty: float = 1.5
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ThompsonConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_size: float = 0.05
    min_position_size: float = 0.005
    daily_loss_limit: float = 0.03
    max_drawdown: float = 0.10
    max_correlated_positions: int = 3
    consecutive_loss_limit: int = 5
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RiskConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """Training configuration."""
    total_episodes: int = 100
    episodes_per_fold: int = 500
    max_steps_per_episode: int = 50
    eval_frequency: int = 25
    checkpoint_frequency: int = 10
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Config:
    """
    Main configuration class providing typed access to all settings.
    """
    
    def __init__(self, config_path: str = None):
        self._raw = load_config(config_path)
        
        # Create typed sub-configs
        self.dqn = DQNConfig.from_dict(self._raw.get('dqn', {}))
        self.thompson = ThompsonConfig.from_dict(self._raw.get('thompson', {}))
        self.risk = RiskConfig.from_dict(self._raw.get('risk', {}))
        self.training = TrainingConfig.from_dict(self._raw.get('training', {}))
    
    @property
    def data(self) -> Dict[str, Any]:
        return self._raw.get('data', {})
    
    @property
    def agent_costs(self) -> Dict[str, float]:
        return self._raw.get('agent_costs', {})
    
    @property
    def reward(self) -> Dict[str, Any]:
        return self._raw.get('reward', {})
    
    @property
    def curriculum(self) -> Dict[str, Any]:
        return self._raw.get('training', {}).get('curriculum', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a raw config value by key."""
        keys = key.split('.')
        value = self._raw
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value


# Global config instance (lazy loaded)
_config: Optional[Config] = None


def get_config(config_path: str = None) -> Config:
    """Get or create the global config instance."""
    global _config
    if _config is None or config_path is not None:
        _config = Config(config_path)
    return _config
