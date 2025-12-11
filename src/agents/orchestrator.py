"""
Cost-Aware Orchestrator for EETA.

Coordinates agent execution with cost awareness.
Key innovation: Run expensive analyses only when necessary.
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np

# from src.agents.historical_agent import HistoricalPatternAgent
# from src.agents.sentiment_agent import SentimentAgent
# from src.agents.market_agent import MarketContextAgent
from .historical_agent import HistoricalPatternAgent
from .sentiment_agent import SentimentAgent
from .market_agent import MarketContextAgent
logger = logging.getLogger(__name__)


class CostAwareOrchestrator:
    """
    Orchestrates agent execution with cost awareness.
    
    Design Principle: Run expensive analyses only when necessary.
    
    Decision Logic:
    1. Always run Historical Agent first (cheap, foundational)
    2. Always run Market Context (cheap, important)
    3. Conditionally run Sentiment Agent based on historical confidence
    4. Aggregate available signals → State vector
    """
    
    def __init__(
        self,
        historical_agent: HistoricalPatternAgent = None,
        sentiment_agent: SentimentAgent = None,
        market_agent: MarketContextAgent = None,
        cost_config: Dict[str, float] = None,
        skip_sentiment_threshold: float = 0.85
    ):
        """
        Initialize orchestrator.
        
        Args:
            historical_agent: Historical pattern analyzer
            sentiment_agent: Sentiment analyzer
            market_agent: Market context analyzer
            cost_config: Cost per agent call
            skip_sentiment_threshold: Confidence above which to skip sentiment
        """
        # Initialize agents
        self.historical_agent = historical_agent or HistoricalPatternAgent()
        self.sentiment_agent = sentiment_agent or SentimentAgent()
        self.market_agent = market_agent or MarketContextAgent()
        
        # Cost configuration
        self.cost_config = cost_config or {
            "historical": 0.1,   # Low cost (local computation)
            "sentiment": 0.5,   # Medium cost (API calls)
            "market": 0.2,      # Low cost (single API call)
        }
        
        # Skip thresholds
        self.skip_sentiment_threshold = skip_sentiment_threshold
        
        # Track agent usage for analysis
        self.agent_calls = {"historical": 0, "sentiment": 0, "market": 0}
        self.agent_skips = {"historical": 0, "sentiment": 0, "market": 0}
        self.total_cost = 0.0
    
    def analyze(
        self,
        ticker: str,
        event_data: Dict[str, Any],
        force_all: bool = False
    ) -> Dict[str, Any]:
        """
        Run analysis with cost-aware agent selection.
        
        Args:
            ticker: Stock symbol
            event_data: Event information including date
            force_all: If True, run all agents regardless of confidence
            
        Returns:
            Dictionary with state, agent_results, and cost
        """
        results = {}
        
        # STEP 1: Always run Historical Agent first (cheap, foundational)
        results["historical"] = self.historical_agent.analyze(ticker, event_data)
        self.agent_calls["historical"] += 1
        self.total_cost += self.cost_config["historical"]
        
        # STEP 2: Always run Market Context (cheap, important)
        results["market"] = self.market_agent.analyze(event_data.get("date"))
        self.agent_calls["market"] += 1
        self.total_cost += self.cost_config["market"]
        
        # STEP 3: Conditionally run Sentiment Agent
        historical_confidence = results["historical"].get("confidence", 0)
        
        if force_all or historical_confidence <= self.skip_sentiment_threshold:
            # Need more signal → Run sentiment analysis
            results["sentiment"] = self.sentiment_agent.analyze(ticker, event_data)
            self.agent_calls["sentiment"] += 1
            self.total_cost += self.cost_config["sentiment"]
        else:
            # High historical confidence → Skip sentiment (save API cost)
            results["sentiment"] = {
                "skipped": True,
                "reason": "high_historical_confidence",
                "news_sentiment": 0.0,  # Neutral imputation
                "news_volume": 0.0,
                "social_sentiment": 0.0,
                "analyst_revision": 0.0,
                "attention_score": 0.0,
                "sentiment_trend": 0.0,
                "dispersion": 0.0,
                "confidence": 0.0
            }
            self.agent_skips["sentiment"] += 1
        
        # STEP 4: Aggregate into state vector
        state = self._aggregate_to_state(results)
        
        return {
            "state": state,
            "agent_results": results,
            "total_cost": self._calculate_analysis_cost(results)
        }
    
    def _aggregate_to_state(self, results: Dict[str, Any]) -> np.ndarray:
        """
        Combine agent outputs into 43-dimensional state vector.
        
        State vector layout:
        - [0-11]: Historical features
        - [12-19]: Sentiment features
        - [20-27]: Market context features
        - [28-33]: Technical features (from event data)
        - [34-35]: Meta features
        """
        # state = np.zeros(36)
        state = np.zeros(43)  # Updated to 43 dimensions
        
        # Historical features (indices 0-11)
        hist = results.get("historical", {})
        state[0] = hist.get("beat_rate", 0.5)
        state[1] = hist.get("avg_move_on_beat", 0.0)
        state[2] = hist.get("avg_move_on_miss", 0.0)
        state[3] = hist.get("move_std", 0.05)
        state[4] = hist.get("consistency", 0.5)
        state[5] = hist.get("guidance_impact", 0.0)
        state[6] = hist.get("post_drift", 0.0)
        state[7] = hist.get("last_surprise", 0.0)
        state[8] = hist.get("surprise_trend", 0.0)
        state[9] = hist.get("beat_streak", 0.0)
        state[10] = hist.get("confidence", 0.5)
        state[11] = hist.get("data_quality", 0.5)
        
        # Sentiment features (indices 12-19)
        sent = results.get("sentiment", {})
        if sent.get("skipped", False):
            # Impute neutral values for skipped sentiment
            state[12:20] = 0.0
        else:
            state[12] = sent.get("news_sentiment", 0.0)
            state[13] = sent.get("news_volume", 0.0)
            state[14] = sent.get("social_sentiment", 0.0)
            state[15] = sent.get("analyst_revision", 0.0)
            state[16] = sent.get("attention_score", 0.0)
            state[17] = sent.get("sentiment_trend", 0.0)
            state[18] = sent.get("dispersion", 0.0)
            state[19] = sent.get("confidence", 0.5)
        
        # Market context features (indices 20-27)
        mkt = results.get("market", {})
        state[20] = mkt.get("vix_normalized", 0.5)
        state[21] = mkt.get("vix_percentile", 0.5)
        state[22] = mkt.get("spy_momentum", 0.0)
        state[23] = mkt.get("sector_momentum", 0.0)
        state[24] = mkt.get("market_regime", 0.0)
        state[25] = mkt.get("breadth", 0.5)
        state[26] = mkt.get("expected_move", 0.05)
        state[27] = mkt.get("confidence", 0.5)
        
        # Technical features (indices 28-33) - from event data or defaults
        tech = results.get("technical", {})
        state[28] = tech.get("rsi_normalized", 0.5)
        state[29] = tech.get("trend_strength", 0.0)
        state[30] = tech.get("volume_ratio", 1.0)
        state[31] = tech.get("gap_risk", 0.0)
        state[32] = tech.get("support_distance", 0.0)
        state[33] = tech.get("momentum", 0.0)
        
        # Meta features (indices 34-35)
        state[34] = self._calculate_signal_agreement(results)
        state[35] = self._calculate_overall_confidence(results)
        
        return state
    
    def _calculate_signal_agreement(self, results: Dict[str, Any]) -> float:
        """Calculate agreement level between different signals."""
        signals = []
        
        # Historical signal direction
        hist = results.get("historical", {})
        beat_rate = hist.get("beat_rate", 0.5)
        if beat_rate > 0.6:
            signals.append(1)  # Bullish signal
        elif beat_rate < 0.4:
            signals.append(-1)  # Bearish signal
        else:
            signals.append(0)  # Neutral
        
        # Market regime signal
        mkt = results.get("market", {})
        regime = mkt.get("market_regime", 0)
        if regime > 0.5:
            signals.append(1)
        elif regime < -0.5:
            signals.append(-1)
        else:
            signals.append(0)
        
        # Sentiment signal
        sent = results.get("sentiment", {})
        if not sent.get("skipped", False):
            sentiment = sent.get("news_sentiment", 0)
            if sentiment > 0.3:
                signals.append(1)
            elif sentiment < -0.3:
                signals.append(-1)
            else:
                signals.append(0)
        
        # Calculate agreement
        if len(signals) == 0:
            return 0.5
        
        # All same direction = high agreement
        if len(set(signals)) == 1:
            return 1.0
        elif sum(signals) == 0 and 0 not in signals:
            return 0.0  # Conflicting signals
        else:
            # Partial agreement
            return 0.3 + 0.4 * (abs(sum(signals)) / len(signals))
    
    def _calculate_overall_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate overall confidence from component confidences."""
        confidences = []
        weights = []
        
        # Historical confidence (highest weight)
        hist_conf = results.get("historical", {}).get("confidence", 0.5)
        confidences.append(hist_conf)
        weights.append(0.5)
        
        # Market confidence
        mkt_conf = results.get("market", {}).get("confidence", 0.9)
        confidences.append(mkt_conf)
        weights.append(0.3)
        
        # Sentiment confidence (if available)
        sent = results.get("sentiment", {})
        if not sent.get("skipped", False):
            sent_conf = sent.get("confidence", 0.3)
            confidences.append(sent_conf)
            weights.append(0.2)
        
        # Weighted average
        total_weight = sum(weights[:len(confidences)])
        if total_weight == 0:
            return 0.5
        
        return sum(c * w for c, w in zip(confidences, weights)) / total_weight
    
    def _calculate_analysis_cost(self, results: Dict[str, Any]) -> float:
        """Calculate total cost for this analysis."""
        cost = self.cost_config["historical"] + self.cost_config["market"]
        
        if not results.get("sentiment", {}).get("skipped", False):
            cost += self.cost_config["sentiment"]
        
        return cost
    
    def get_efficiency_stats(self) -> Dict[str, Any]:
        """Return agent usage statistics for analysis."""
        total_calls = sum(self.agent_calls.values())
        total_skips = sum(self.agent_skips.values())
        
        return {
            "agent_calls": self.agent_calls.copy(),
            "agent_skips": self.agent_skips.copy(),
            "total_analyses": self.agent_calls["historical"],  # One per analysis
            "efficiency_ratio": total_skips / max(1, total_calls + total_skips),
            "total_cost": self.total_cost,
            "estimated_cost_savings": self.agent_skips["sentiment"] * self.cost_config["sentiment"]
        }
    
    def reset_stats(self):
        """Reset usage statistics."""
        self.agent_calls = {"historical": 0, "sentiment": 0, "market": 0}
        self.agent_skips = {"historical": 0, "sentiment": 0, "market": 0}
        self.total_cost = 0.0
    
    def set_market_cache(self, vix_data=None, spy_data=None):
        """Set market data cache for faster analysis."""
        self.market_agent.set_cache(vix_data, spy_data)


class SimpleOrchestrator:
    """
    Simple orchestrator that always runs all agents.
    
    Useful for comparison in ablation studies.
    """
    
    def __init__(
        self,
        historical_agent: HistoricalPatternAgent = None,
        sentiment_agent: SentimentAgent = None,
        market_agent: MarketContextAgent = None
    ):
        self.historical_agent = historical_agent or HistoricalPatternAgent()
        self.sentiment_agent = sentiment_agent or SentimentAgent()
        self.market_agent = market_agent or MarketContextAgent()
    
    def analyze(self, ticker: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run all agents."""
        results = {
            "historical": self.historical_agent.analyze(ticker, event_data),
            "sentiment": self.sentiment_agent.analyze(ticker, event_data),
            "market": self.market_agent.analyze(event_data.get("date"))
        }
        
        # Use same aggregation as CostAwareOrchestrator
        orchestrator = CostAwareOrchestrator()
        state = orchestrator._aggregate_to_state(results)
        
        return {
            "state": state,
            "agent_results": results
        }
