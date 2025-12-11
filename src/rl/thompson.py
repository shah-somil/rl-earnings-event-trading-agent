"""
Thompson Sampling for Position Sizing.

Implements Thompson Sampling with Beta distributions for
dynamic position size selection under uncertainty.

Key Benefits:
- Naturally handles exploration vs exploitation
- Adapts position sizes based on historical success
- Context-aware adjustments for confidence and volatility
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy import stats
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ThompsonSampler:
    """
    Thompson Sampling for position size selection.
    
    Uses Beta distributions to model success probability of each
    position size bucket. Samples from posteriors to select sizes.
    
    Position Size Buckets: [0.5%, 1%, 2%, 3%, 5%]
    """
    
    def __init__(
        self,
        size_buckets: List[float] = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        confidence_weight: float = 2.0,
        volatility_penalty: float = 1.5
    ):
        """
        Initialize Thompson Sampler.
        
        Args:
            size_buckets: Position size options (as decimals)
            prior_alpha: Prior successes (Beta parameter)
            prior_beta: Prior failures (Beta parameter)
            confidence_weight: Weight for confidence adjustment
            volatility_penalty: Penalty for high volatility
        """
        if size_buckets is None:
            size_buckets = [0.005, 0.01, 0.02, 0.03, 0.05]  # 0.5% to 5%
        
        self.size_buckets = np.array(size_buckets)
        self.num_buckets = len(size_buckets)
        
        # Context adjustment parameters
        self.confidence_weight = confidence_weight
        self.volatility_penalty = volatility_penalty
        
        # Initialize Beta distribution parameters
        # Each bucket has (alpha, beta) for successes and failures
        self.alphas = np.ones(self.num_buckets) * prior_alpha
        self.betas = np.ones(self.num_buckets) * prior_beta
        
        # Tracking
        self.history = []  # (bucket_idx, reward, context)
        self.selections = np.zeros(self.num_buckets, dtype=int)
    
    def select_size(
        self,
        confidence: float = 0.5,
        volatility: float = 0.0,
        greedy: bool = False
    ) -> Tuple[int, float]:
        """
        Select position size using Thompson Sampling.
        
        Args:
            confidence: Signal confidence (0-1)
            volatility: Market volatility (0-1 normalized)
            greedy: If True, select highest expected value
            
        Returns:
            Tuple of (bucket_index, position_size)
        """
        # Get context-adjusted parameters
        adj_alphas, adj_betas = self._adjust_for_context(confidence, volatility)
        
        if greedy:
            # Select bucket with highest expected value
            expected_values = adj_alphas / (adj_alphas + adj_betas)
            bucket_idx = np.argmax(expected_values)
        else:
            # Thompson Sampling: sample from each posterior
            samples = np.array([
                np.random.beta(a, b) for a, b in zip(adj_alphas, adj_betas)
            ])
            bucket_idx = np.argmax(samples)
        
        # Track selection
        self.selections[bucket_idx] += 1
        
        return bucket_idx, self.size_buckets[bucket_idx]
    
    def _adjust_for_context(
        self,
        confidence: float,
        volatility: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adjust Beta parameters based on context.
        
        High confidence → Favor larger sizes
        High volatility → Favor smaller sizes (risk management)
        
        Args:
            confidence: Signal confidence (0-1)
            volatility: Market volatility (0-1)
            
        Returns:
            Tuple of adjusted (alphas, betas)
        """
        # Confidence shift: positive for high confidence
        confidence_shift = (confidence - 0.5) * self.confidence_weight
        
        # Volatility shift: negative (penalizes larger sizes)
        volatility_shift = volatility * self.volatility_penalty
        
        # Create size-dependent adjustments
        # Larger buckets get more boost from confidence, more penalty from volatility
        size_weights = np.linspace(0, 1, self.num_buckets)
        
        # Adjust alphas (success counts)
        adj_alphas = self.alphas.copy()
        adj_alphas += confidence_shift * size_weights
        
        # Adjust betas (failure counts)
        adj_betas = self.betas.copy()
        adj_betas += volatility_shift * size_weights
        
        # Ensure positive values
        adj_alphas = np.maximum(0.1, adj_alphas)
        adj_betas = np.maximum(0.1, adj_betas)
        
        return adj_alphas, adj_betas
    
    def update(
        self,
        bucket_idx: int,
        reward: float,
        profit_threshold: float = 0.0
    ):
        """
        Update Beta distribution based on outcome.
        
        Args:
            bucket_idx: Index of selected bucket
            reward: Actual reward/P&L received
            profit_threshold: Threshold for "success"
        """
        if bucket_idx < 0 or bucket_idx >= self.num_buckets:
            logger.warning(f"Invalid bucket index: {bucket_idx}")
            return
        
        # Binary outcome: profit or loss
        if reward > profit_threshold:
            self.alphas[bucket_idx] += 1  # Success
        else:
            self.betas[bucket_idx] += 1   # Failure
        
        # Track history
        self.history.append({
            'bucket_idx': bucket_idx,
            'size': self.size_buckets[bucket_idx],
            'reward': reward,
            'success': reward > profit_threshold
        })
    
    def get_expected_values(self) -> np.ndarray:
        """Get expected success probability for each bucket."""
        return self.alphas / (self.alphas + self.betas)
    
    def get_uncertainties(self) -> np.ndarray:
        """Get uncertainty (variance) for each bucket."""
        total = self.alphas + self.betas
        return (self.alphas * self.betas) / (total ** 2 * (total + 1))
    
    def get_confidence_intervals(
        self,
        confidence_level: float = 0.95
    ) -> List[Tuple[float, float]]:
        """
        Get confidence intervals for each bucket.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95)
            
        Returns:
            List of (lower, upper) bounds
        """
        intervals = []
        alpha_level = (1 - confidence_level) / 2
        
        for a, b in zip(self.alphas, self.betas):
            lower = stats.beta.ppf(alpha_level, a, b)
            upper = stats.beta.ppf(1 - alpha_level, a, b)
            intervals.append((lower, upper))
        
        return intervals
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sampler statistics."""
        return {
            'num_buckets': self.num_buckets,
            'size_buckets': self.size_buckets.tolist(),
            'alphas': self.alphas.tolist(),
            'betas': self.betas.tolist(),
            'expected_values': self.get_expected_values().tolist(),
            'selections': self.selections.tolist(),
            'total_updates': len(self.history),
        }
    
    def reset(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """Reset to prior distributions."""
        self.alphas = np.ones(self.num_buckets) * prior_alpha
        self.betas = np.ones(self.num_buckets) * prior_beta
        self.history = []
        self.selections = np.zeros(self.num_buckets, dtype=int)
    
    def save(self, path: str):
        """Save sampler state to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'size_buckets': self.size_buckets.tolist(),
            'alphas': self.alphas.tolist(),
            'betas': self.betas.tolist(),
            'selections': self.selections.tolist(),
            'confidence_weight': self.confidence_weight,
            'volatility_penalty': self.volatility_penalty,
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Thompson Sampler saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ThompsonSampler':
        """Load sampler state from file."""
        with open(path, 'r') as f:
            state = json.load(f)
        
        sampler = cls(
            size_buckets=state['size_buckets'],
            confidence_weight=state.get('confidence_weight', 2.0),
            volatility_penalty=state.get('volatility_penalty', 1.5)
        )
        
        sampler.alphas = np.array(state['alphas'])
        sampler.betas = np.array(state['betas'])
        sampler.selections = np.array(state['selections'])
        
        logger.info(f"Thompson Sampler loaded from {path}")
        return sampler


class ContextualThompsonSampler(ThompsonSampler):
    """
    Extended Thompson Sampler with richer context modeling.
    
    Maintains separate Beta distributions for different context buckets
    (e.g., different market regimes).
    """
    
    def __init__(
        self,
        size_buckets: List[float] = None,
        num_context_buckets: int = 3,  # e.g., bearish, neutral, bullish
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        **kwargs
    ):
        """
        Initialize contextual sampler.
        
        Args:
            size_buckets: Position size options
            num_context_buckets: Number of context categories
            prior_alpha: Prior alpha
            prior_beta: Prior beta
            **kwargs: Additional parameters
        """
        super().__init__(size_buckets, prior_alpha, prior_beta, **kwargs)
        
        self.num_context_buckets = num_context_buckets
        
        # Separate distributions per context
        # Shape: (num_context_buckets, num_size_buckets)
        self.context_alphas = np.ones((num_context_buckets, self.num_buckets)) * prior_alpha
        self.context_betas = np.ones((num_context_buckets, self.num_buckets)) * prior_beta
    
    def select_size_contextual(
        self,
        context_idx: int,
        confidence: float = 0.5,
        volatility: float = 0.0,
        greedy: bool = False
    ) -> Tuple[int, float]:
        """
        Select position size based on context category.
        
        Args:
            context_idx: Context bucket index (e.g., 0=bearish, 1=neutral, 2=bullish)
            confidence: Signal confidence
            volatility: Market volatility
            greedy: If True, use greedy selection
            
        Returns:
            Tuple of (bucket_index, position_size)
        """
        context_idx = np.clip(context_idx, 0, self.num_context_buckets - 1)
        
        alphas = self.context_alphas[context_idx]
        betas = self.context_betas[context_idx]
        
        # Apply confidence/volatility adjustments
        adj_alphas, adj_betas = self._adjust_params(
            alphas, betas, confidence, volatility
        )
        
        if greedy:
            expected_values = adj_alphas / (adj_alphas + adj_betas)
            bucket_idx = np.argmax(expected_values)
        else:
            samples = np.array([
                np.random.beta(a, b) for a, b in zip(adj_alphas, adj_betas)
            ])
            bucket_idx = np.argmax(samples)
        
        self.selections[bucket_idx] += 1
        return bucket_idx, self.size_buckets[bucket_idx]
    
    def _adjust_params(
        self,
        alphas: np.ndarray,
        betas: np.ndarray,
        confidence: float,
        volatility: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Adjust parameters for context."""
        confidence_shift = (confidence - 0.5) * self.confidence_weight
        volatility_shift = volatility * self.volatility_penalty
        size_weights = np.linspace(0, 1, self.num_buckets)
        
        adj_alphas = alphas + confidence_shift * size_weights
        adj_betas = betas + volatility_shift * size_weights
        
        return np.maximum(0.1, adj_alphas), np.maximum(0.1, adj_betas)
    
    def update_contextual(
        self,
        context_idx: int,
        bucket_idx: int,
        reward: float,
        profit_threshold: float = 0.0
    ):
        """Update distribution for specific context."""
        context_idx = np.clip(context_idx, 0, self.num_context_buckets - 1)
        
        if reward > profit_threshold:
            self.context_alphas[context_idx, bucket_idx] += 1
        else:
            self.context_betas[context_idx, bucket_idx] += 1
        
        # Also update global counts
        super().update(bucket_idx, reward, profit_threshold)


def confidence_to_bucket_hint(confidence: float, num_buckets: int = 5) -> int:
    """
    Map confidence to a suggested position size bucket.
    
    Used as a heuristic baseline for comparison with Thompson Sampling.
    
    Args:
        confidence: Signal confidence (0-1)
        num_buckets: Number of size buckets
        
    Returns:
        Suggested bucket index
    """
    # Low confidence → small size, high confidence → larger size
    return min(num_buckets - 1, int(confidence * num_buckets))
