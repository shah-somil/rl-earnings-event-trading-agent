"""
Market Context Agent for EETA.

Captures macro market context using freely available data:
- VIX level and percentile
- SPY momentum
- Market regime
- Sector performance
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketContextAgent:
    """
    Captures macro market context.
    
    All data freely available via yfinance:
    - VIX level and percentile
    - SPY momentum
    - Market regime
    - Sector performance
    """
    
    def __init__(self):
        """Initialize market context agent."""
        # Cache for market data
        self._vix_cache = None
        self._spy_cache = None
        self._cache_date = None
    
    def analyze(
        self,
        as_of_date: Any,
        vix_data: pd.DataFrame = None,
        spy_data: pd.DataFrame = None
    ) -> Dict[str, float]:
        """
        Analyze market context as of a specific date.
        
        Args:
            as_of_date: Date for analysis (enables backtesting)
            vix_data: Optional pre-fetched VIX data
            spy_data: Optional pre-fetched SPY data
            
        Returns:
            Dictionary of market features
        """
        # Convert date if needed
        if isinstance(as_of_date, str):
            as_of_date = pd.to_datetime(as_of_date)
        elif isinstance(as_of_date, date) and not isinstance(as_of_date, datetime):
            as_of_date = datetime.combine(as_of_date, datetime.min.time())
        
        # Fetch or use provided data
        vix = vix_data if vix_data is not None else self._vix_cache
        spy = spy_data if spy_data is not None else self._spy_cache
        
        # Analyze VIX
        vix_features = self._analyze_vix(vix, as_of_date)
        
        # Analyze SPY
        spy_features = self._analyze_spy(spy, as_of_date)
        
        # Classify market regime
        market_regime = self._classify_regime(spy_features, vix_features)
        
        return {
            'vix_normalized': vix_features['normalized'],
            'vix_percentile': vix_features['percentile'],
            'spy_momentum': spy_features['momentum'],
            'sector_momentum': 0.0,  # Can extend to sector-specific
            'market_regime': market_regime,
            'breadth': 0.5,  # Placeholder
            'expected_move': self._calculate_expected_move(vix_features['current']),
            'confidence': 0.9  # Market data is reliable
        }
    
    def _analyze_vix(
        self,
        vix_data: Optional[pd.DataFrame],
        as_of_date: datetime
    ) -> Dict[str, float]:
        """Analyze VIX data."""
        result = {
            'current': 20.0,  # Default VIX
            'normalized': 0.5,
            'percentile': 0.5
        }
        
        if vix_data is None or vix_data.empty:
            return result
        
        try:
            # Ensure timezone compatibility
            if hasattr(vix_data.index, 'tz') and vix_data.index.tz is not None:
                if as_of_date.tzinfo is None:
                    as_of_date = as_of_date.tz_localize(vix_data.index.tz)
            
            # Filter to data before as_of_date
            mask = vix_data.index <= as_of_date
            vix_hist = vix_data[mask]
            
            if len(vix_hist) == 0:
                return result
            
            # Get current VIX
            current = vix_hist['close'].iloc[-1] if 'close' in vix_hist.columns else 20.0
            
            # Calculate percentile over last year
            year_data = vix_hist.tail(252)
            if len(year_data) > 0 and 'close' in year_data.columns:
                percentile = (year_data['close'] < current).mean()
            else:
                percentile = 0.5
            
            result['current'] = current
            result['normalized'] = self._normalize_vix(current)
            result['percentile'] = percentile
            
        except Exception as e:
            logger.debug(f"VIX analysis error: {e}")
        
        return result
    
    def _analyze_spy(
        self,
        spy_data: Optional[pd.DataFrame],
        as_of_date: datetime
    ) -> Dict[str, float]:
        """Analyze SPY data."""
        result = {
            'momentum': 0.0,
            'trend': 0.0,
            'volatility': 0.5
        }
        
        if spy_data is None or spy_data.empty:
            return result
        
        try:
            # Ensure timezone compatibility
            if hasattr(spy_data.index, 'tz') and spy_data.index.tz is not None:
                if as_of_date.tzinfo is None:
                    as_of_date = as_of_date.tz_localize(spy_data.index.tz)
            
            # Filter to data before as_of_date
            mask = spy_data.index <= as_of_date
            spy_hist = spy_data[mask]
            
            if len(spy_hist) < 20:
                return result
            
            close = spy_hist['close'] if 'close' in spy_hist.columns else spy_hist.iloc[:, 0]
            
            # 20-day momentum
            if len(close) >= 20:
                momentum = close.iloc[-1] / close.iloc[-20] - 1
                result['momentum'] = np.clip(momentum, -0.2, 0.2)
            
            # Trend (above or below 50-day MA)
            if len(close) >= 50:
                ma50 = close.rolling(50).mean().iloc[-1]
                result['trend'] = 1 if close.iloc[-1] > ma50 else -1
            
            # Volatility (20-day realized)
            if len(close) >= 20:
                returns = close.pct_change().dropna()
                result['volatility'] = returns.tail(20).std() * np.sqrt(252)
            
        except Exception as e:
            logger.debug(f"SPY analysis error: {e}")
        
        return result
    
    def _normalize_vix(self, vix: float) -> float:
        """
        Normalize VIX to 0-1 range.
        
        VIX typically ranges 10-80:
        - 10-15: Very low volatility
        - 15-20: Normal
        - 20-30: Elevated
        - 30+: High fear
        """
        return np.clip((vix - 10) / 70, 0, 1)
    
    def _calculate_expected_move(self, vix: float, days: int = 1) -> float:
        """
        Calculate expected move based on VIX.
        
        This is our key formula for simulating options behavior.
        Expected Move = VIX/100 * sqrt(days/365)
        
        For a typical earnings event (1 day):
        - VIX 15: ~0.8% expected move
        - VIX 20: ~1.0% expected move
        - VIX 30: ~1.6% expected move
        """
        return (vix / 100) * np.sqrt(days / 365)
    
    def _classify_regime(
        self,
        spy_features: Dict[str, float],
        vix_features: Dict[str, float]
    ) -> float:
        """
        Classify market regime.
        
        Returns: -1 (bearish) to +1 (bullish)
        """
        momentum = spy_features.get('momentum', 0)
        vix_level = vix_features.get('current', 20)
        
        # High VIX + negative momentum = bearish
        # Low VIX + positive momentum = bullish
        if momentum > 0.05 and vix_level < 20:
            return 1.0  # Bullish
        elif momentum < -0.05 and vix_level > 25:
            return -1.0  # Bearish
        elif momentum > 0.02 and vix_level < 25:
            return 0.5  # Slightly bullish
        elif momentum < -0.02 and vix_level > 20:
            return -0.5  # Slightly bearish
        else:
            return 0.0  # Neutral
    
    def set_cache(
        self,
        vix_data: pd.DataFrame = None,
        spy_data: pd.DataFrame = None
    ):
        """Set cached market data."""
        if vix_data is not None:
            self._vix_cache = vix_data
        if spy_data is not None:
            self._spy_cache = spy_data
        self._cache_date = datetime.now()
    
    def clear_cache(self):
        """Clear cached data."""
        self._vix_cache = None
        self._spy_cache = None
        self._cache_date = None


def get_market_regime_label(regime_value: float) -> str:
    """Convert numeric regime to label."""
    if regime_value >= 0.5:
        return 'BULLISH'
    elif regime_value <= -0.5:
        return 'BEARISH'
    else:
        return 'NEUTRAL'
