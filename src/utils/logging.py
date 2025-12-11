"""
Logging configuration for EETA.

Provides consistent logging across all modules with support for
file and console output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    name: str = "eeta",
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_str: str = None
) -> logging.Logger:
    """
    Set up logging for a module.
    
    Args:
        name: Logger name (usually module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        format_str: Custom format string
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Default format
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_str)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Module name (use __name__ for automatic naming)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TrainingLogger:
    """
    Specialized logger for training metrics.
    
    Tracks and formats training progress, metrics, and checkpoints.
    """
    
    def __init__(self, experiment_name: str = None, log_dir: str = "experiments/logs"):
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up file logger
        log_file = self.log_dir / f"{self.experiment_name}.log"
        self.logger = setup_logging(
            name=f"eeta.training.{self.experiment_name}",
            level="DEBUG",
            log_file=str(log_file)
        )
        
        # Metrics storage
        self.metrics_history = []
        self.episode_count = 0
    
    def log_episode(
        self,
        episode: int,
        reward: float,
        loss: float = None,
        epsilon: float = None,
        pnl: float = None,
        extra: dict = None
    ):
        """Log metrics for a training episode."""
        self.episode_count = episode
        
        metrics = {
            "episode": episode,
            "reward": reward,
            "loss": loss,
            "epsilon": epsilon,
            "pnl": pnl,
            **(extra or {})
        }
        self.metrics_history.append(metrics)
        
        # Format log message
        parts = [f"Episode {episode:4d}"]
        parts.append(f"Reward: {reward:+.4f}")
        if loss is not None:
            parts.append(f"Loss: {loss:.6f}")
        if epsilon is not None:
            parts.append(f"ε: {epsilon:.3f}")
        if pnl is not None:
            parts.append(f"P&L: {pnl:+.2%}")
        
        self.logger.info(" | ".join(parts))
    
    def log_fold(self, fold_id: int, train_years: str, test_year: int):
        """Log start of a new validation fold."""
        self.logger.info("=" * 60)
        self.logger.info(f"FOLD {fold_id}: Train {train_years}, Test {test_year}")
        self.logger.info("=" * 60)
    
    def log_evaluation(self, metrics: dict):
        """Log evaluation metrics."""
        self.logger.info("-" * 40)
        self.logger.info("EVALUATION RESULTS:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
        self.logger.info("-" * 40)
    
    def log_checkpoint(self, path: str, metrics: dict = None):
        """Log checkpoint save."""
        self.logger.info(f"Checkpoint saved: {path}")
        if metrics:
            self.logger.info(f"  Best metrics: {metrics}")
    
    def log_warning(self, message: str):
        """Log a warning."""
        self.logger.warning(message)
    
    def log_error(self, message: str, exc_info: bool = False):
        """Log an error."""
        self.logger.error(message, exc_info=exc_info)
    
    def get_metrics_df(self):
        """Get metrics history as a DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.metrics_history)


# Module-level logger
_module_logger = None


def get_module_logger() -> logging.Logger:
    """Get the main EETA module logger."""
    global _module_logger
    if _module_logger is None:
        _module_logger = setup_logging("eeta", level="INFO")
    return _module_logger
