"""
Historical Pattern Agent for EETA.

Analyzes historical earnings patterns for a given stock.
Features extracted include beat rate, average moves, consistency, etc.
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HistoricalPatternAgent:
    """
    Analyzes historical earnings patterns for a given stock.
    
    Features extracted:
    - Beat/miss rate
    - Average move magnitude
    - Reaction consistency
    - Guidance patterns
    - Recent trend
    """
    
    def __init__(
        self,
        lookback_quarters: int = 12,
        min_quarters: int = 4
    ):
        """
        Initialize agent.
        
        Args:
            lookback_quarters: Number of past earnings to analyze
            min_quarters: Minimum quarters needed for analysis
        """
        self.lookback = lookback_quarters
        self.min_quarters = min_quarters
        
        # Cache for earnings history
        self._cache = {}
    
    def analyze(
        self,
        ticker: str,
        event_data: Dict[str, Any],
        earnings_history: pd.DataFrame = None
    ) -> Dict[str, float]:
        """
        Analyze historical earnings patterns.
        
        Args:
            ticker: Stock symbol
            event_data: Contains earnings_date
            earnings_history: Optional pre-fetched earnings history
            
        Returns:
            Dictionary of historical features
        """
        # Get earnings history
        if earnings_history is not None:
            history = earnings_history
        else:
            history = self._get_or_fetch_history(ticker)
        
        if history is None or len(history) < self.min_quarters:
            return self._insufficient_data_response()
        
        # Filter to data before current event
        current_date = event_data.get('date', datetime.now())
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        
        # Only use historical data (before this event)
        history = self._filter_before_date(history, current_date)
        
        if len(history) < self.min_quarters:
            return self._insufficient_data_response()
        
        # Calculate features
        beat_rate = self._calculate_beat_rate(history)
        move_stats = self._calculate_move_statistics(history)
        consistency = self._calculate_consistency(history)
        guidance = self._analyze_guidance_patterns(history)
        trend = self._analyze_recent_trend(history)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            n_quarters=len(history),
            consistency=consistency,
            data_completeness=self._assess_data_quality(history)
        )
        
        return {
            'beat_rate': beat_rate,
            'avg_move_on_beat': move_stats['beat_avg'],
            'avg_move_on_miss': move_stats['miss_avg'],
            'move_std': move_stats['std'],
            'consistency': consistency,
            'guidance_impact': guidance['impact'],
            'post_drift': move_stats['drift'],
            'last_surprise': trend['last_surprise'],
            'surprise_trend': trend['trend'],
            'beat_streak': trend['streak'],
            'confidence': confidence,
            'data_quality': self._assess_data_quality(history)
        }
    
    def _get_or_fetch_history(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get cached or fetch earnings history."""
        if ticker in self._cache:
            return self._cache[ticker]
        
        # Would fetch from data source - for now return None
        # In production: self._cache[ticker] = fetch_earnings_history(ticker)
        return None
    
    def _filter_before_date(
        self,
        history: pd.DataFrame,
        date: datetime
    ) -> pd.DataFrame:
        """Filter history to only include events before date."""
        if 'date' in history.columns:
            date_col = 'date'
        elif 'earnings_date' in history.columns:
            date_col = 'earnings_date'
        else:
            return history
        
        history[date_col] = pd.to_datetime(history[date_col])
        return history[history[date_col] < date].tail(self.lookback)
    
    def _calculate_beat_rate(self, history: pd.DataFrame) -> float:
        """Calculate historical beat rate."""
        if 'eps_actual' not in history.columns or 'eps_estimate' not in history.columns:
            return 0.5
        
        valid = history.dropna(subset=['eps_actual', 'eps_estimate'])
        if len(valid) == 0:
            return 0.5
        
        beats = (valid['eps_actual'] > valid['eps_estimate']).sum()
        return beats / len(valid)
    
    def _calculate_move_statistics(self, history: pd.DataFrame) -> Dict[str, float]:
        """Calculate move statistics."""
        result = {
            'beat_avg': 0.0,
            'miss_avg': 0.0,
            'std': 0.05,
            'drift': 0.0
        }
        
        if 'actual_move' not in history.columns:
            if 'move_pct' in history.columns:
                history = history.rename(columns={'move_pct': 'actual_move'})
            else:
                return result
        
        # Separate beats and misses
        if 'eps_actual' in history.columns and 'eps_estimate' in history.columns:
            beats = history[history['eps_actual'] > history['eps_estimate']]
            misses = history[history['eps_actual'] <= history['eps_estimate']]
            
            if len(beats) > 0 and 'actual_move' in beats.columns:
                result['beat_avg'] = beats['actual_move'].mean()
            if len(misses) > 0 and 'actual_move' in misses.columns:
                result['miss_avg'] = misses['actual_move'].mean()
        
        # Overall statistics
        if 'actual_move' in history.columns:
            moves = history['actual_move'].dropna()
            if len(moves) > 0:
                result['std'] = moves.std()
                # Post-earnings drift (last few quarters)
                recent = moves.tail(4)
                result['drift'] = recent.mean() if len(recent) > 0 else 0.0
        
        return result
    
    def _calculate_consistency(self, history: pd.DataFrame) -> float:
        """
        Calculate earnings reaction consistency.
        
        High consistency = stock reacts predictably to beats/misses
        Low consistency = reactions are noisy
        """
        if 'actual_move' not in history.columns:
            return 0.5
        
        if 'eps_actual' not in history.columns or 'eps_estimate' not in history.columns:
            return 0.5
        
        # Check if beats consistently lead to up moves
        beats = history[history['eps_actual'] > history['eps_estimate']]
        misses = history[history['eps_actual'] <= history['eps_estimate']]
        
        beat_up_rate = 0.5
        miss_down_rate = 0.5
        
        if len(beats) > 0 and 'actual_move' in beats.columns:
            beat_up_rate = (beats['actual_move'] > 0).mean()
        
        if len(misses) > 0 and 'actual_move' in misses.columns:
            miss_down_rate = (misses['actual_move'] < 0).mean()
        
        # Consistency is average of these directional accuracies
        consistency = (beat_up_rate + miss_down_rate) / 2
        
        # Adjust for sample size
        n = len(history)
        if n < 8:
            consistency = 0.5 + (consistency - 0.5) * (n / 8)
        
        return consistency
    
    def _analyze_guidance_patterns(self, history: pd.DataFrame) -> Dict[str, float]:
        """Analyze guidance impact (placeholder)."""
        # Would need guidance data from earnings calls
        return {'impact': 0.0}
    
    def _analyze_recent_trend(self, history: pd.DataFrame) -> Dict[str, Any]:
        """Analyze recent earnings trend."""
        result = {
            'last_surprise': 0.0,
            'trend': 0.0,
            'streak': 0.0
        }
        
        if 'surprise_pct' in history.columns:
            surprises = history['surprise_pct'].dropna()
            if len(surprises) > 0:
                result['last_surprise'] = surprises.iloc[-1]
                
                # Trend: are surprises improving or deteriorating?
                if len(surprises) >= 4:
                    recent = surprises.tail(4).mean()
                    older = surprises.head(len(surprises) - 4).mean() if len(surprises) > 4 else 0
                    result['trend'] = np.clip(recent - older, -1, 1)
        
        # Calculate beat streak
        if 'eps_actual' in history.columns and 'eps_estimate' in history.columns:
            valid = history.dropna(subset=['eps_actual', 'eps_estimate'])
            if len(valid) > 0:
                beats = (valid['eps_actual'] > valid['eps_estimate']).astype(int)
                # Count consecutive beats/misses from most recent
                streak = 0
                for b in reversed(beats.values):
                    if b == beats.iloc[-1]:
                        streak += 1 if b else -1
                    else:
                        break
                result['streak'] = np.clip(streak, -4, 4)
        
        return result
    
    def _calculate_confidence(
        self,
        n_quarters: int,
        consistency: float,
        data_completeness: float
    ) -> float:
        """
        Calculate confidence in historical signal.
        
        High confidence when:
        - More data points
        - Higher consistency
        - Complete data
        """
        # More quarters → higher confidence (diminishing returns)
        quarters_factor = min(1.0, n_quarters / self.lookback)
        
        # Combine factors
        confidence = (
            0.4 * quarters_factor +
            0.4 * consistency +
            0.2 * data_completeness
        )
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _assess_data_quality(self, history: pd.DataFrame) -> float:
        """Assess data completeness and quality."""
        if len(history) == 0:
            return 0.0
        
        required_cols = ['eps_actual', 'eps_estimate', 'actual_move']
        available = sum(1 for c in required_cols if c in history.columns)
        col_score = available / len(required_cols)
        
        # Check for missing values
        missing_rate = history.isnull().mean().mean()
        completeness = 1 - missing_rate
        
        return (col_score + completeness) / 2
    
    def _insufficient_data_response(self) -> Dict[str, float]:
        """Return default values when data is insufficient."""
        return {
            'beat_rate': 0.5,
            'avg_move_on_beat': 0.0,
            'avg_move_on_miss': 0.0,
            'move_std': 0.05,
            'consistency': 0.5,
            'guidance_impact': 0.0,
            'post_drift': 0.0,
            'last_surprise': 0.0,
            'surprise_trend': 0.0,
            'beat_streak': 0.0,
            'confidence': 0.2,  # Low confidence due to insufficient data
            'data_quality': 0.0
        }
    
    def set_cache(self, ticker: str, history: pd.DataFrame):
        """Set cached earnings history for a ticker."""
        self._cache[ticker] = history
    
    def clear_cache(self):
        """Clear the earnings history cache."""
        self._cache.clear()
