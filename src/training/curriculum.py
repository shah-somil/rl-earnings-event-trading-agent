"""
Curriculum Learning for EETA.

Implements curriculum learning to train the agent progressively
on easier to harder examples.

Difficulty levels:
- Easy: Large-cap, predictable stocks (AAPL, MSFT, etc.)
- Medium: S&P 500 large-caps with more variation
- Hard: All stocks including mid-caps with less predictability
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default easy tickers (mega-cap, predictable)
DEFAULT_EASY_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
    "NVDA", "TSLA", "JPM", "JNJ", "V"
]

# Medium tickers (large-cap with more variation)
DEFAULT_MEDIUM_TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM",
    "ADBE", "NFLX", "PYPL", "CSCO", "ORCL",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "AXP",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "HD", "MCD", "NKE"
]


class DifficultyLevel:
    """Difficulty level enum."""
    EASY = 'easy'
    MEDIUM = 'medium'
    HARD = 'hard'


class CurriculumScheduler:
    """
    Schedules curriculum progression based on training progress.
    
    Progression strategies:
    - Episode-based: Fixed number of episodes per level
    - Performance-based: Advance when performance threshold met
    - Adaptive: Combination of both
    """
    
    def __init__(
        self,
        easy_episodes: int = 30,
        medium_episodes: int = 40,
        hard_episodes: int = 30,
        performance_threshold: float = 0.6,
        strategy: str = 'episode'
    ):
        """
        Initialize curriculum scheduler.
        
        Args:
            easy_episodes: Episodes for easy level
            medium_episodes: Episodes for medium level
            hard_episodes: Episodes for hard level
            performance_threshold: Win rate to advance (for adaptive)
            strategy: 'episode', 'performance', or 'adaptive'
        """
        self.easy_episodes = easy_episodes
        self.medium_episodes = medium_episodes
        self.hard_episodes = hard_episodes
        self.performance_threshold = performance_threshold
        self.strategy = strategy
        
        self.current_level = DifficultyLevel.EASY
        self.episode_count = 0
        self.level_start_episode = 0
        
        # Performance tracking
        self.level_performance = []
    
    def get_current_level(self) -> str:
        """Get current difficulty level."""
        return self.current_level
    
    def step(self, episode: int, performance: float = None) -> str:
        """
        Update curriculum based on episode and performance.
        
        Args:
            episode: Current episode number
            performance: Optional performance metric (e.g., win rate)
            
        Returns:
            Current difficulty level
        """
        self.episode_count = episode
        
        if performance is not None:
            self.level_performance.append(performance)
        
        # Check for level advancement
        if self.strategy == 'episode':
            self._check_episode_advancement()
        elif self.strategy == 'performance':
            self._check_performance_advancement()
        else:  # adaptive
            self._check_adaptive_advancement()
        
        return self.current_level
    
    def _check_episode_advancement(self):
        """Check advancement based on episode count."""
        episodes_in_level = self.episode_count - self.level_start_episode
        
        if self.current_level == DifficultyLevel.EASY:
            if episodes_in_level >= self.easy_episodes:
                self._advance_level()
        elif self.current_level == DifficultyLevel.MEDIUM:
            if episodes_in_level >= self.medium_episodes:
                self._advance_level()
        # Hard level doesn't advance
    
    def _check_performance_advancement(self):
        """Check advancement based on performance."""
        if len(self.level_performance) < 10:
            return
        
        recent_perf = np.mean(self.level_performance[-10:])
        
        if recent_perf >= self.performance_threshold:
            self._advance_level()
    
    def _check_adaptive_advancement(self):
        """Check advancement using adaptive strategy."""
        episodes_in_level = self.episode_count - self.level_start_episode
        
        # Minimum episodes per level
        min_episodes = {
            DifficultyLevel.EASY: self.easy_episodes // 2,
            DifficultyLevel.MEDIUM: self.medium_episodes // 2,
            DifficultyLevel.HARD: float('inf')
        }
        
        if episodes_in_level < min_episodes[self.current_level]:
            return
        
        # Can advance based on performance OR episode count
        episode_ready = episodes_in_level >= {
            DifficultyLevel.EASY: self.easy_episodes,
            DifficultyLevel.MEDIUM: self.medium_episodes,
            DifficultyLevel.HARD: float('inf')
        }[self.current_level]
        
        perf_ready = False
        if len(self.level_performance) >= 10:
            recent_perf = np.mean(self.level_performance[-10:])
            perf_ready = recent_perf >= self.performance_threshold
        
        if episode_ready or perf_ready:
            self._advance_level()
    
    def _advance_level(self):
        """Advance to next difficulty level."""
        if self.current_level == DifficultyLevel.EASY:
            self.current_level = DifficultyLevel.MEDIUM
            logger.info(f"Curriculum: Advanced to MEDIUM at episode {self.episode_count}")
        elif self.current_level == DifficultyLevel.MEDIUM:
            self.current_level = DifficultyLevel.HARD
            logger.info(f"Curriculum: Advanced to HARD at episode {self.episode_count}")
        
        self.level_start_episode = self.episode_count
        self.level_performance = []
    
    def reset(self):
        """Reset curriculum to beginning."""
        self.current_level = DifficultyLevel.EASY
        self.episode_count = 0
        self.level_start_episode = 0
        self.level_performance = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get curriculum statistics."""
        return {
            'current_level': self.current_level,
            'episode_count': self.episode_count,
            'episodes_in_level': self.episode_count - self.level_start_episode,
            'avg_level_performance': np.mean(self.level_performance) if self.level_performance else 0.0
        }


class CurriculumDataSelector:
    """
    Selects training data based on curriculum difficulty level.
    """
    
    def __init__(
        self,
        full_data: pd.DataFrame,
        ticker_column: str = 'ticker',
        easy_tickers: List[str] = None,
        medium_tickers: List[str] = None
    ):
        """
        Initialize data selector.
        
        Args:
            full_data: Complete dataset
            ticker_column: Name of ticker column
            easy_tickers: List of easy tickers
            medium_tickers: List of medium tickers (superset of easy)
        """
        self.full_data = full_data.copy()
        self.ticker_column = ticker_column
        
        self.easy_tickers = set(easy_tickers or DEFAULT_EASY_TICKERS)
        self.medium_tickers = set(medium_tickers or DEFAULT_MEDIUM_TICKERS)
        
        # All tickers for hard level
        self.all_tickers = set(full_data[ticker_column].unique())
        
        logger.info(f"Curriculum data: {len(self.easy_tickers)} easy, "
                   f"{len(self.medium_tickers)} medium, {len(self.all_tickers)} total tickers")
    
    def get_data_for_level(self, level: str) -> pd.DataFrame:
        """
        Get training data for a difficulty level.
        
        Args:
            level: Difficulty level ('easy', 'medium', 'hard')
            
        Returns:
            Filtered DataFrame
        """
        if level == DifficultyLevel.EASY:
            tickers = self.easy_tickers
        elif level == DifficultyLevel.MEDIUM:
            tickers = self.medium_tickers
        else:  # HARD
            tickers = self.all_tickers
        
        mask = self.full_data[self.ticker_column].isin(tickers)
        filtered = self.full_data[mask].copy()
        
        logger.debug(f"Level {level}: {len(filtered)} samples from {len(tickers)} tickers")
        
        return filtered
    
    def get_difficulty_label(self, ticker: str) -> str:
        """Get difficulty label for a ticker."""
        if ticker in self.easy_tickers:
            return DifficultyLevel.EASY
        elif ticker in self.medium_tickers:
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.HARD
    
    def add_difficulty_column(self) -> pd.DataFrame:
        """Add difficulty column to full dataset."""
        data = self.full_data.copy()
        data['difficulty'] = data[self.ticker_column].apply(self.get_difficulty_label)
        return data


class CurriculumManager:
    """
    Combined curriculum scheduler and data selector.
    
    Provides a simple interface for curriculum learning.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any] = None
    ):
        """
        Initialize curriculum manager.
        
        Args:
            data: Full training dataset
            config: Curriculum configuration
        """
        config = config or {}
        
        self.scheduler = CurriculumScheduler(
            easy_episodes=config.get('easy_episodes', 30),
            medium_episodes=config.get('medium_episodes', 40),
            hard_episodes=config.get('hard_episodes', 30),
            performance_threshold=config.get('performance_threshold', 0.6),
            strategy=config.get('strategy', 'episode')
        )
        
        self.selector = CurriculumDataSelector(
            full_data=data,
            ticker_column=config.get('ticker_column', 'ticker'),
            easy_tickers=config.get('easy_tickers'),
            medium_tickers=config.get('medium_tickers')
        )
    
    def get_current_data(self) -> pd.DataFrame:
        """Get training data for current difficulty level."""
        level = self.scheduler.get_current_level()
        return self.selector.get_data_for_level(level)
    
    def step(self, episode: int, performance: float = None) -> pd.DataFrame:
        """
        Update curriculum and return current data.
        
        Args:
            episode: Current episode number
            performance: Optional performance metric
            
        Returns:
            Training data for current level
        """
        self.scheduler.step(episode, performance)
        return self.get_current_data()
    
    def get_level(self) -> str:
        """Get current difficulty level."""
        return self.scheduler.get_current_level()
    
    def reset(self):
        """Reset curriculum."""
        self.scheduler.reset()
