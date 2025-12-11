"""
Sentiment Analysis Agent for EETA.

Analyzes market sentiment from news and social sources.
Uses free APIs when available (Finnhub), otherwise provides
neutral defaults.
"""

import logging
import os
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


# Simple lexicon for basic sentiment analysis
POSITIVE_WORDS = {
    'beat', 'beats', 'exceeded', 'exceeds', 'strong', 'growth', 'profit',
    'gain', 'gains', 'up', 'surge', 'surges', 'rally', 'rallies', 'bullish',
    'positive', 'optimistic', 'outperform', 'upgrade', 'raised', 'better',
    'record', 'high', 'boost', 'accelerate', 'momentum', 'breakthrough'
}

NEGATIVE_WORDS = {
    'miss', 'misses', 'missed', 'below', 'weak', 'decline', 'loss', 'losses',
    'down', 'fall', 'falls', 'drop', 'drops', 'bearish', 'negative', 'pessimistic',
    'underperform', 'downgrade', 'cut', 'worse', 'low', 'slump', 'slowdown',
    'warning', 'concern', 'risk', 'uncertain'
}


class SimpleSentimentAnalyzer:
    """
    Simple lexicon-based sentiment analyzer.
    
    Used as fallback when FinBERT or other models are unavailable.
    """
    
    def __init__(self):
        self.positive_words = POSITIVE_WORDS
        self.negative_words = NEGATIVE_WORDS
    
    def score(self, text: str) -> float:
        """
        Score text sentiment from -1 (negative) to +1 (positive).
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score
        """
        if not text:
            return 0.0
        
        # Tokenize and clean
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            return 0.0
        
        # Count positive and negative words
        pos_count = sum(1 for w in words if w in self.positive_words)
        neg_count = sum(1 for w in words if w in self.negative_words)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        # Score: (pos - neg) / total, capped at [-1, 1]
        score = (pos_count - neg_count) / total
        return np.clip(score, -1.0, 1.0)
    
    def score_batch(self, texts: List[str]) -> List[float]:
        """Score multiple texts."""
        return [self.score(t) for t in texts]


class SentimentAgent:
    """
    Analyzes market sentiment from news and social sources.
    
    Uses free APIs:
    - Finnhub for news (if API key available)
    - Simple lexicon-based sentiment
    """
    
    def __init__(
        self,
        finnhub_api_key: str = None,
        lookback_days: int = 7
    ):
        """
        Initialize sentiment agent.
        
        Args:
            finnhub_api_key: API key for Finnhub
            lookback_days: Days of news to analyze
        """
        self.finnhub_key = finnhub_api_key or os.getenv("FINNHUB_API_KEY")
        self.lookback_days = lookback_days
        
        # Initialize sentiment analyzer
        self.analyzer = SimpleSentimentAnalyzer()
        
        # Try to initialize Finnhub client
        self.finnhub_client = None
        if self.finnhub_key:
            try:
                import finnhub
                self.finnhub_client = finnhub.Client(api_key=self.finnhub_key)
                logger.info("Finnhub client initialized")
            except ImportError:
                logger.warning("finnhub package not installed")
            except Exception as e:
                logger.warning(f"Failed to initialize Finnhub: {e}")
    
    def analyze(
        self,
        ticker: str,
        event_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Analyze pre-earnings sentiment.
        
        Args:
            ticker: Stock symbol
            event_data: Contains earnings_date
            
        Returns:
            Dictionary of sentiment features
        """
        earnings_date = event_data.get('date', datetime.now())
        if isinstance(earnings_date, str):
            earnings_date = datetime.strptime(earnings_date, '%Y-%m-%d')
        
        # Try to fetch news
        news = self._fetch_news(ticker, earnings_date)
        
        if not news:
            return self._no_news_response()
        
        # Analyze sentiment
        sentiments = [self.analyzer.score(article) for article in news]
        
        # Get analyst data (if available)
        analyst_data = self._fetch_analyst_revisions(ticker)
        
        return {
            'news_sentiment': np.mean(sentiments),
            'news_volume': self._normalize_volume(len(news)),
            'social_sentiment': 0.0,  # Placeholder if no social API
            'analyst_revision': analyst_data.get('revision_score', 0.0),
            'attention_score': self._calculate_attention(len(news)),
            'sentiment_trend': self._calculate_trend(sentiments),
            'dispersion': np.std(sentiments) if len(sentiments) > 1 else 0.0,
            'confidence': self._calculate_confidence(len(news), np.std(sentiments) if len(sentiments) > 1 else 0.5)
        }
    
    def _fetch_news(
        self,
        ticker: str,
        before_date: datetime
    ) -> List[str]:
        """Fetch news articles from Finnhub."""
        if not self.finnhub_client:
            return []
        
        try:
            end = before_date
            start = end - timedelta(days=self.lookback_days)
            
            news = self.finnhub_client.company_news(
                ticker,
                _from=start.strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d")
            )
            
            # Extract headlines
            headlines = []
            for article in (news or []):
                headline = article.get('headline', '')
                summary = article.get('summary', '')
                # Combine headline and summary for better analysis
                text = f"{headline} {summary}".strip()
                if text:
                    headlines.append(text)
            
            return headlines
            
        except Exception as e:
            logger.warning(f"Finnhub fetch failed for {ticker}: {e}")
            return []
    
    def _fetch_analyst_revisions(
        self,
        ticker: str,
        lookback_days: int = 30
    ) -> Dict[str, float]:
        """Fetch analyst revision data."""
        if not self.finnhub_client:
            return {'revision_score': 0.0}
        
        try:
            # Get recommendation trends
            recs = self.finnhub_client.recommendation_trends(ticker)
            
            if not recs or len(recs) == 0:
                return {'revision_score': 0.0}
            
            # Most recent recommendation summary
            latest = recs[0]
            
            # Calculate score based on buy/sell recommendations
            buy = latest.get('buy', 0) + latest.get('strongBuy', 0)
            sell = latest.get('sell', 0) + latest.get('strongSell', 0)
            hold = latest.get('hold', 0)
            
            total = buy + sell + hold
            if total == 0:
                return {'revision_score': 0.0}
            
            # Score: (buy - sell) / total
            score = (buy - sell) / total
            
            return {'revision_score': np.clip(score, -1.0, 1.0)}
            
        except Exception as e:
            logger.debug(f"Analyst data fetch failed for {ticker}: {e}")
            return {'revision_score': 0.0}
    
    def _normalize_volume(self, count: int, max_count: int = 50) -> float:
        """Normalize news count to 0-1 range."""
        return min(1.0, count / max_count)
    
    def _calculate_attention(self, news_count: int) -> float:
        """Calculate attention score based on news volume."""
        # More news = more attention
        # Normalize: 0 news = 0, 10+ news = high attention
        return min(1.0, news_count / 10)
    
    def _calculate_trend(self, sentiments: List[float]) -> float:
        """Calculate sentiment trend (improving or deteriorating)."""
        if len(sentiments) < 3:
            return 0.0
        
        # Split into earlier and later
        mid = len(sentiments) // 2
        early = np.mean(sentiments[:mid])
        late = np.mean(sentiments[mid:])
        
        return np.clip(late - early, -1.0, 1.0)
    
    def _calculate_confidence(
        self,
        news_count: int,
        sentiment_std: float
    ) -> float:
        """
        Calculate confidence in sentiment signal.
        
        High confidence when:
        - More news articles
        - Lower sentiment dispersion (agreement)
        """
        # Volume factor
        volume_factor = min(1.0, news_count / 20)
        
        # Agreement factor (lower std = higher agreement)
        agreement_factor = 1 - min(1.0, sentiment_std)
        
        confidence = 0.6 * volume_factor + 0.4 * agreement_factor
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _no_news_response(self) -> Dict[str, float]:
        """Return default values when no news available."""
        return {
            'news_sentiment': 0.0,
            'news_volume': 0.0,
            'social_sentiment': 0.0,
            'analyst_revision': 0.0,
            'attention_score': 0.0,
            'sentiment_trend': 0.0,
            'dispersion': 0.0,
            'confidence': 0.1  # Very low confidence without news
        }
    
    def is_available(self) -> bool:
        """Check if sentiment analysis is available."""
        return self.finnhub_client is not None
