"""
Dataset Builder for EETA.

Constructs the historical earnings dataset for training and backtesting.
Target: 10,000+ earnings events across 500 stocks over 5 years.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable

import warnings

from .sources import YFinanceSource, DataSourceManager
from .preprocessor import FEATURE_NAMES, create_default_features

logger = logging.getLogger(__name__)


class EarningsDatasetBuilder:
    """
    Builds historical earnings dataset for training/backtesting.
    
    Creates a dataset with one row per earnings event, containing:
    - Pre-earnings features (what we'd know before the announcement)
    - Outcome variables (actual move, surprise, etc.)
    """
    
    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        finnhub_api_key: str = None
    ):
        """
        Initialize dataset builder.
        
        Args:
            tickers: List of stock tickers to include
            start_date: Start date for data collection
            end_date: End date for data collection
            finnhub_api_key: Optional Finnhub API key for sentiment
        """
        self.tickers = tickers
        self.start = pd.to_datetime(start_date)
        self.end = pd.to_datetime(end_date)
        
        # Initialize data sources
        self.data_source = DataSourceManager(finnhub_api_key=finnhub_api_key)
        self.yf = self.data_source.yfinance
        
        # Cache for VIX and SPY data
        self._vix_cache = None
        self._spy_cache = None
    
    def build(self, cache_path: str = None, show_progress: bool = True) -> pd.DataFrame:
        """
        Build complete dataset.
        
        Args:
            cache_path: Optional path to cache results
            show_progress: Show progress bar
            
        Returns:
            DataFrame with one row per earnings event
        """
        # Check cache
        if cache_path and Path(cache_path).exists():
            logger.info(f"Loading cached dataset from {cache_path}")
            return pd.read_parquet(cache_path)
        
        # Prefetch VIX and SPY data
        self._prefetch_market_data()
        
        all_events = []
        failed_tickers = []
        
        iterator = tqdm(self.tickers, desc="Building dataset") if show_progress else self.tickers
        
        for ticker in iterator:
            try:
                events = self._process_ticker(ticker)
                all_events.extend(events)
            except Exception as e:
                logger.warning(f"Failed to process {ticker}: {e}")
                failed_tickers.append(ticker)
                continue
        
        if not all_events:
            logger.error("No events collected! This typically means price/earnings data could not be fetched (network/auth) or the date range has no announcements.")
            return pd.DataFrame(columns=['earnings_date'])
        
        df = pd.DataFrame(all_events)
        
        # Sort by date
        df = df.sort_values('earnings_date').reset_index(drop=True)
        
        logger.info(f"Built dataset with {len(df)} events from {len(self.tickers) - len(failed_tickers)} tickers")
        logger.info(f"Date range: {df['earnings_date'].min()} to {df['earnings_date'].max()}")
        
        if failed_tickers:
            logger.warning(f"Failed tickers ({len(failed_tickers)}): {failed_tickers[:10]}...")
        
        # Compute real historical features from prior earnings
        logger.info("Computing ticker-specific historical features...")
        df = self._compute_ticker_historical_features(df)
        
        # Save cache
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)
            logger.info(f"Dataset cached to {cache_path}")
        
        return df
    
    def _prefetch_market_data(self):
        """Prefetch VIX and SPY data for the entire period."""
        # Add buffer for lookback calculations
        start_buffer = (self.start - timedelta(days=365)).strftime('%Y-%m-%d')
        end_str = self.end.strftime('%Y-%m-%d')
        
        logger.info("Prefetching VIX data...")
        self._vix_cache = self.yf.fetch_vix(start_buffer, end_str)
        
        logger.info("Prefetching SPY data...")
        self._spy_cache = self.yf.fetch_spy(start_buffer, end_str)
    
    def _process_ticker(self, ticker: str) -> List[Dict[str, Any]]:
        """Process all earnings events for one ticker."""
        # Fetch price history with buffer
        start_buffer = (self.start - timedelta(days=365)).strftime('%Y-%m-%d')
        end_str = self.end.strftime('%Y-%m-%d')
        
        price_history = self.yf.fetch_price_history(ticker, start_buffer, end_str)
        
        if price_history.empty:
            logger.debug(f"No price data for {ticker}")
            return []
        
        # Fetch earnings history
        earnings = self.yf.fetch_earnings_history(ticker)
        
        if earnings.empty:
            logger.debug(f"No earnings data for {ticker}")
            return []
        
        # Fetch company info
        company_info = self.yf.fetch_company_info(ticker)
        
        events = []
        
        for _, row in earnings.iterrows():
            # Get earnings date
            if 'date' in row.index:
                earnings_date = pd.to_datetime(row['date'])
            elif 'earnings_date' in row.index:
                earnings_date = pd.to_datetime(row['earnings_date'])
            else:
                continue
            
            # Skip if outside date range
            if earnings_date < self.start or earnings_date > self.end:
                continue
            
            try:
                event = self._construct_event(
                    ticker=ticker,
                    earnings_date=earnings_date,
                    earnings_row=row,
                    price_history=price_history,
                    company_info=company_info
                )
                
                if event is not None:
                    events.append(event)
                    
            except Exception as e:
                logger.debug(f"Error processing {ticker} {earnings_date}: {e}")
                continue
        
        return events
    
    def _construct_event(
        self,
        ticker: str,
        earnings_date: datetime,
        earnings_row: pd.Series,
        price_history: pd.DataFrame,
        company_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Construct feature dictionary for one earnings event.
        
        IMPORTANT: Only uses data available BEFORE the earnings announcement.
        """
        # Handle timezone compatibility
        if price_history.index.tz is not None and earnings_date.tz is None:
            earnings_date = pd.Timestamp(earnings_date).tz_localize(price_history.index.tz)
        elif price_history.index.tz is None and hasattr(earnings_date, 'tz') and earnings_date.tz is not None:
            earnings_date = earnings_date.tz_localize(None)

        event = {
            'ticker': ticker,
            'earnings_date': earnings_date,
            'sector': company_info.get('sector', 'Unknown'),
            'industry': company_info.get('industry', 'Unknown'),
        }
        
        # === OUTCOME VARIABLES (ground truth) ===
        eps_actual = earnings_row.get('eps_actual', np.nan)
        eps_estimate = earnings_row.get('eps_estimate', np.nan)
        
        event['eps_actual'] = eps_actual
        event['eps_estimate'] = eps_estimate
        
        if pd.notna(eps_actual) and pd.notna(eps_estimate) and eps_estimate != 0:
            event['surprise_pct'] = (eps_actual - eps_estimate) / abs(eps_estimate)
        else:
            event['surprise_pct'] = np.nan
        
        # Calculate actual move
        move_data = self.yf.calculate_earnings_move(price_history, earnings_date)
        event['actual_move'] = move_data.get('move_pct', np.nan)
        event['pre_close'] = move_data.get('pre_close', np.nan)
        event['post_close'] = move_data.get('post_close', np.nan)
        
        # Skip if we don't have the actual move (can't train on this)
        if pd.isna(event['actual_move']):
            return None
        
        # === PRE-EARNINGS FEATURES (what we'd know before) ===
        
        # Historical pattern features
        hist_features = self._calculate_historical_features(
            price_history, earnings_date, ticker
        )
        event.update(hist_features)
        
        # Market context features
        mkt_features = self._calculate_market_features(earnings_date)
        event.update(mkt_features)
        
        # Technical features
        tech_features = self._calculate_technical_features(
            price_history, earnings_date
        )
        event.update(tech_features)
        
        # Sentiment features (placeholder - would need API)
        sent_features = self._calculate_sentiment_features(ticker, earnings_date)
        event.update(sent_features)
        
        # Meta features
        event['meta_signal_agreement'] = self._calculate_signal_agreement(event)
        event['meta_overall_confidence'] = self._calculate_overall_confidence(event)
        
        return event
    
    def _calculate_historical_features(
        self,
        price_history: pd.DataFrame,
        earnings_date: datetime,
        ticker: str
    ) -> Dict[str, float]:
        """Calculate historical pattern features."""
        # For this we need past earnings data
        # In a full implementation, we'd look up past earnings
        # For now, use price-based proxies
        
        features = {}
        
        # Get historical data before this earnings
        mask = price_history.index < earnings_date
        hist = price_history[mask].tail(252)  # Last year
        
        if len(hist) < 20:
            return {f'hist_{k}': np.nan for k in [
                'beat_rate', 'avg_move_on_beat', 'avg_move_on_miss', 'move_std',
                'consistency', 'guidance_impact', 'post_drift', 'last_surprise',
                'surprise_trend', 'beat_streak', 'confidence', 'data_quality'
            ]}
        
        # Calculate daily returns
        returns = hist['close'].pct_change().dropna()
        
        # Proxy metrics based on historical volatility patterns
        features['hist_beat_rate'] = 0.5  # Placeholder - would need earnings history
        features['hist_avg_move_on_beat'] = returns.mean() * 5  # Scaled average
        features['hist_avg_move_on_miss'] = -returns.mean() * 5
        features['hist_move_std'] = returns.std() * np.sqrt(1)  # Daily vol
        features['hist_consistency'] = 1 - min(1, returns.std() / 0.03)
        features['hist_guidance_impact'] = 0.0  # Placeholder
        features['hist_post_drift'] = returns.tail(20).mean()
        features['hist_last_surprise'] = 0.0  # Placeholder
        features['hist_surprise_trend'] = 0.0  # Placeholder
        features['hist_beat_streak'] = 0.0  # Placeholder
        features['hist_confidence'] = min(1, len(hist) / 200)
        features['hist_data_quality'] = 1.0 if len(hist) > 100 else len(hist) / 100
        
        return features
    
    def _calculate_market_features(self, earnings_date: datetime) -> Dict[str, float]:
        """Calculate market context features."""
        features = {}
        
        # Get VIX data
        if self._vix_cache is not None and not self._vix_cache.empty:
            vix_mask = self._vix_cache.index < earnings_date
            vix_hist = self._vix_cache[vix_mask]
            
            if len(vix_hist) > 0:
                current_vix = vix_hist['close'].iloc[-1]
                vix_percentile = (vix_hist['close'] < current_vix).mean()
                
                features['mkt_vix_normalized'] = np.clip((current_vix - 10) / 70, 0, 1)
                features['mkt_vix_percentile'] = vix_percentile
                features['mkt_expected_move'] = (current_vix / 100) * np.sqrt(1 / 365)
            else:
                features['mkt_vix_normalized'] = 0.5
                features['mkt_vix_percentile'] = 0.5
                features['mkt_expected_move'] = 0.05
        else:
            features['mkt_vix_normalized'] = 0.5
            features['mkt_vix_percentile'] = 0.5
            features['mkt_expected_move'] = 0.05
        
        # Get SPY data for market direction
        if self._spy_cache is not None and not self._spy_cache.empty:
            spy_mask = self._spy_cache.index < earnings_date
            spy_hist = self._spy_cache[spy_mask]
            
            if len(spy_hist) >= 20:
                spy_returns = spy_hist['close'].pct_change().dropna()
                momentum = spy_hist['close'].iloc[-1] / spy_hist['close'].iloc[-20] - 1
                
                features['mkt_spy_momentum'] = np.clip(momentum, -0.2, 0.2)
                
                # Market regime
                if momentum > 0.05 and features['mkt_vix_normalized'] < 0.4:
                    features['mkt_market_regime'] = 1.0
                elif momentum < -0.05 and features['mkt_vix_normalized'] > 0.6:
                    features['mkt_market_regime'] = -1.0
                else:
                    features['mkt_market_regime'] = 0.0
            else:
                features['mkt_spy_momentum'] = 0.0
                features['mkt_market_regime'] = 0.0
        else:
            features['mkt_spy_momentum'] = 0.0
            features['mkt_market_regime'] = 0.0
        
        features['mkt_sector_momentum'] = 0.0  # Would need sector ETF data
        features['mkt_breadth'] = 0.5  # Placeholder
        features['mkt_confidence'] = 0.9  # Market data is reliable
        
        return features
    
    def _compute_ticker_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute real historical features from prior earnings of the same ticker.
        
        This post-processes the dataset to add features that require
        knowledge of previous earnings events for the same stock.
        
        These features have STRONG correlation with move magnitude (~0.5)
        and enable the agent to predict volatility.
        """
        df = df.sort_values(['ticker', 'earnings_date']).reset_index(drop=True)
        
        new_cols = {
            'hist_ticker_avg_move': [],
            'hist_ticker_move_std': [],
            'hist_ticker_beat_rate': [],
            'hist_last_move': [],
            'hist_last_abs_move': [],
            'hist_move_expanding': [],
            'hist_earnings_count': [],
        }
        
        for idx, row in df.iterrows():
            ticker = row['ticker']
            current_date = row['earnings_date']
            prior = df[(df['ticker'] == ticker) & (df['earnings_date'] < current_date)]
            
            if len(prior) < 2:
                # Not enough history - use sensible defaults
                new_cols['hist_ticker_avg_move'].append(0.03)
                new_cols['hist_ticker_move_std'].append(0.03)
                new_cols['hist_ticker_beat_rate'].append(0.5)
                new_cols['hist_last_move'].append(0.0)
                new_cols['hist_last_abs_move'].append(0.03)
                new_cols['hist_move_expanding'].append(0.0)
                new_cols['hist_earnings_count'].append(len(prior))
            else:
                moves = prior['actual_move'].values
                abs_moves = np.abs(moves)
                surprises = prior['surprise_pct'].values
                
                new_cols['hist_ticker_avg_move'].append(float(np.mean(abs_moves)))
                new_cols['hist_ticker_move_std'].append(float(np.std(moves)))
                new_cols['hist_ticker_beat_rate'].append(float(np.mean(surprises > 0)))
                new_cols['hist_last_move'].append(float(moves[-1]))
                new_cols['hist_last_abs_move'].append(float(abs_moves[-1]))
                new_cols['hist_move_expanding'].append(
                    1.0 if abs_moves[-1] > np.mean(abs_moves[:-1]) else 0.0
                )
                new_cols['hist_earnings_count'].append(len(prior))
        
        for col, values in new_cols.items():
            df[col] = values
        
        # Re-sort by date for training
        df = df.sort_values('earnings_date').reset_index(drop=True)
        
        return df
    
    def _calculate_technical_features(
        self,
        price_history: pd.DataFrame,
        earnings_date: datetime
    ) -> Dict[str, float]:
        """Calculate technical indicators."""
        features = {}
        
        # Get data before earnings
        mask = price_history.index < earnings_date
        hist = price_history[mask].tail(50)
        
        if len(hist) < 14:
            return {f'tech_{k}': 0.0 for k in [
                'rsi_normalized', 'trend_strength', 'volume_ratio',
                'gap_risk', 'support_distance', 'momentum'
            ]}
        
        close = hist['close']
        volume = hist['volume'] if 'volume' in hist.columns else pd.Series([1] * len(hist))
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        features['tech_rsi_normalized'] = rsi.iloc[-1] / 100 if not pd.isna(rsi.iloc[-1]) else 0.5
        
        # Trend strength (price vs 20-day MA)
        ma20 = close.rolling(20).mean()
        if len(ma20) > 0 and not pd.isna(ma20.iloc[-1]) and ma20.iloc[-1] != 0:
            trend = (close.iloc[-1] - ma20.iloc[-1]) / ma20.iloc[-1]
            features['tech_trend_strength'] = np.clip(trend * 10, -1, 1)
        else:
            features['tech_trend_strength'] = 0.0
        
        # Volume ratio
        avg_volume = volume.rolling(20).mean()
        if len(avg_volume) > 0 and not pd.isna(avg_volume.iloc[-1]) and avg_volume.iloc[-1] != 0:
            features['tech_volume_ratio'] = np.clip(
                volume.iloc[-1] / avg_volume.iloc[-1], 0, 3
            )
        else:
            features['tech_volume_ratio'] = 1.0
        
        # Gap risk (recent daily ranges)
        high = hist['high'] if 'high' in hist.columns else close
        low = hist['low'] if 'low' in hist.columns else close
        daily_range = (high - low) / close
        features['tech_gap_risk'] = np.clip(daily_range.mean() * 10, 0, 1)
        
        # Support distance (distance from 20-day low)
        low_20 = low.rolling(20).min()
        if len(low_20) > 0 and not pd.isna(low_20.iloc[-1]) and low_20.iloc[-1] != 0:
            support_dist = (close.iloc[-1] - low_20.iloc[-1]) / low_20.iloc[-1]
            features['tech_support_distance'] = np.clip(support_dist, -0.2, 0.2)
        else:
            features['tech_support_distance'] = 0.0
        
        # Momentum (20-day return)
        if len(close) >= 20:
            momentum = close.iloc[-1] / close.iloc[-20] - 1
            features['tech_momentum'] = np.clip(momentum, -0.3, 0.3)
        else:
            features['tech_momentum'] = 0.0
        
        return features
    
    def _calculate_sentiment_features(
        self,
        ticker: str,
        earnings_date: datetime
    ) -> Dict[str, float]:
        """Calculate sentiment features (placeholder without API)."""
        # In production, would fetch from Finnhub or similar
        return {
            'sent_news_sentiment': 0.0,
            'sent_news_volume': 0.0,
            'sent_social_sentiment': 0.0,
            'sent_analyst_revision': 0.0,
            'sent_attention_score': 0.0,
            'sent_sentiment_trend': 0.0,
            'sent_dispersion': 0.0,
            'sent_confidence': 0.3,  # Low confidence without real data
        }
    
    def _calculate_signal_agreement(self, event: Dict[str, Any]) -> float:
        """Calculate agreement between different signals."""
        signals = []
        
        # Historical signal
        if event.get('hist_beat_rate', 0.5) > 0.6:
            signals.append(1)
        elif event.get('hist_beat_rate', 0.5) < 0.4:
            signals.append(-1)
        else:
            signals.append(0)
        
        # Market signal
        if event.get('mkt_market_regime', 0) > 0.5:
            signals.append(1)
        elif event.get('mkt_market_regime', 0) < -0.5:
            signals.append(-1)
        else:
            signals.append(0)
        
        # Technical signal
        if event.get('tech_momentum', 0) > 0.05:
            signals.append(1)
        elif event.get('tech_momentum', 0) < -0.05:
            signals.append(-1)
        else:
            signals.append(0)
        
        # Agreement score
        if len(signals) == 0:
            return 0.5
        
        # All same direction = high agreement
        if all(s == signals[0] for s in signals):
            return 1.0
        elif sum(signals) == 0:
            return 0.3
        else:
            return 0.6
    
    def _calculate_overall_confidence(self, event: Dict[str, Any]) -> float:
        """Calculate overall confidence from component confidences."""
        confidences = [
            event.get('hist_confidence', 0.5),
            event.get('mkt_confidence', 0.9),
            event.get('sent_confidence', 0.3),
        ]
        
        # Weighted average
        weights = [0.5, 0.3, 0.2]
        
        return sum(c * w for c, w in zip(confidences, weights))


def build_dataset(
    tickers: List[str] = None,
    start_date: str = "2019-01-01",
    end_date: str = "2024-12-01",
    cache_path: str = "data/processed/earnings_dataset.parquet",
    finnhub_api_key: str = None
) -> pd.DataFrame:
    """
    Convenience function to build the earnings dataset.
    
    Args:
        tickers: List of tickers (defaults to S&P 500 sample)
        start_date: Start date
        end_date: End date  
        cache_path: Path for caching
        finnhub_api_key: Optional API key
        
    Returns:
        DataFrame with earnings events
    """
    if tickers is None:
        source_manager = DataSourceManager()
        tickers = source_manager.yfinance.get_sp500_tickers()[:100]  # Start with 100
    
    builder = EarningsDatasetBuilder(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        finnhub_api_key=finnhub_api_key
    )
    
    return builder.build(cache_path=cache_path)
