"""
Feature Preprocessing for EETA.

Handles normalization, scaling, and transformation of features
for the DQN state space.

CRITICAL: Scalers must be fitted ONLY on training data to prevent look-ahead bias.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


# Feature specifications: (name, range, description)
FEATURE_SPECS = {
    # Historical features (0-11)
    'hist_beat_rate': (0, 1, 'Historical beat rate'),
    'hist_avg_move_on_beat': (-0.3, 0.3, 'Average move when beating'),
    'hist_avg_move_on_miss': (-0.3, 0.3, 'Average move when missing'),
    'hist_move_std': (0, 0.2, 'Standard deviation of moves'),
    'hist_consistency': (0, 1, 'Prediction consistency'),
    'hist_guidance_impact': (-1, 1, 'Guidance impact score'),
    'hist_post_drift': (-0.1, 0.1, 'Post-earnings drift'),
    'hist_last_surprise': (-0.5, 0.5, 'Most recent surprise'),
    'hist_surprise_trend': (-1, 1, 'Trend in surprises'),
    'hist_beat_streak': (-4, 4, 'Consecutive beats/misses'),
    'hist_confidence': (0, 1, 'Historical signal confidence'),
    'hist_data_quality': (0, 1, 'Data completeness score'),
    
    # Ticker-specific historical features (high predictive power)
    'hist_ticker_avg_move': (0, 0.2, 'Ticker avg earnings move magnitude'),
    'hist_ticker_move_std': (0, 0.2, 'Ticker earnings move volatility'),
    'hist_ticker_beat_rate': (0, 1, 'Ticker historical beat rate'),
    'hist_last_move': (-0.3, 0.3, 'Last earnings move'),
    'hist_last_abs_move': (0, 0.3, 'Last earnings move magnitude'),
    'hist_move_expanding': (0, 1, 'Is volatility expanding'),
    'hist_earnings_count': (0, 20, 'Number of prior earnings'), 

    # Sentiment features (12-19)
    'sent_news_sentiment': (-1, 1, 'News sentiment score'),
    'sent_news_volume': (0, 1, 'Normalized news volume'),
    'sent_social_sentiment': (-1, 1, 'Social media sentiment'),
    'sent_analyst_revision': (-1, 1, 'Analyst revision direction'),
    'sent_attention_score': (0, 1, 'Market attention level'),
    'sent_sentiment_trend': (-1, 1, 'Sentiment change trend'),
    'sent_dispersion': (0, 1, 'Sentiment disagreement'),
    'sent_confidence': (0, 1, 'Sentiment signal confidence'),
    
    # Market context features (20-27)
    'mkt_vix_normalized': (0, 1, 'Normalized VIX level'),
    'mkt_vix_percentile': (0, 1, 'VIX percentile rank'),
    'mkt_spy_momentum': (-0.2, 0.2, 'SPY 20-day momentum'),
    'mkt_sector_momentum': (-0.2, 0.2, 'Sector momentum'),
    'mkt_market_regime': (-1, 1, 'Market regime indicator'),
    'mkt_breadth': (0, 1, 'Market breadth'),
    'mkt_expected_move': (0, 0.2, 'VIX-implied expected move'),
    'mkt_confidence': (0, 1, 'Market signal confidence'),
    
    # Technical features (28-33)
    'tech_rsi_normalized': (0, 1, 'Normalized RSI'),
    'tech_trend_strength': (-1, 1, 'Trend direction/strength'),
    'tech_volume_ratio': (0, 3, 'Volume vs average'),
    'tech_gap_risk': (0, 1, 'Gap risk indicator'),
    'tech_support_distance': (-0.2, 0.2, 'Distance from support'),
    'tech_momentum': (-0.3, 0.3, 'Price momentum'),
    
    # Meta features (34-35)
    'meta_signal_agreement': (0, 1, 'Agent signal agreement'),
    'meta_overall_confidence': (0, 1, 'Combined confidence'),
}

FEATURE_NAMES = list(FEATURE_SPECS.keys())
STATE_DIM = len(FEATURE_NAMES)

assert STATE_DIM == 43, f"Expected 43 features, got {STATE_DIM}"


class StatePreprocessor:
    """
    Preprocesses raw features into normalized state vector.
    
    CRITICAL: Must be fitted only on training data to prevent look-ahead bias.
    
    Features are normalized using StandardScaler and clipped to [-3, 3].
    """
    
    def __init__(self, clip_range: Tuple[float, float] = (-3.0, 3.0)):
        self.clip_range = clip_range
        self.scalers: Dict[str, StandardScaler] = {}
        self.fitted = False
        self._feature_names = FEATURE_NAMES.copy()
    
    def fit(self, training_data: pd.DataFrame) -> 'StatePreprocessor':
        """
        Fit scalers on training data only.
        
        IMPORTANT: This must be called BEFORE processing any test data.
        
        Args:
            training_data: DataFrame with feature columns
            
        Returns:
            Self for chaining
        """
        logger.info(f"Fitting preprocessor on {len(training_data)} samples")
        
        for feature_name in self._feature_names:
            if feature_name in training_data.columns:
                scaler = StandardScaler()
                values = training_data[[feature_name]].dropna()
                
                if len(values) > 0:
                    scaler.fit(values)
                    self.scalers[feature_name] = scaler
                else:
                    logger.warning(f"No valid data for feature: {feature_name}")
                    self.scalers[feature_name] = self._default_scaler()
            else:
                logger.warning(f"Feature not found in data: {feature_name}")
                self.scalers[feature_name] = self._default_scaler()
        
        self.fitted = True
        return self
    
    def _default_scaler(self) -> StandardScaler:
        """Create a default scaler (identity transform)."""
        scaler = StandardScaler()
        # Fit on [0] to create identity-like transform
        scaler.fit([[0.0]])
        scaler.mean_ = np.array([0.0])
        scaler.scale_ = np.array([1.0])
        return scaler
    
    def transform(self, features: Dict[str, float]) -> np.ndarray:
        """
        Transform features to normalized state vector.
        
        Args:
            features: Dictionary of raw feature values
            
        Returns:
            43-dimensional numpy array
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit() first.")
        
        state = np.zeros(STATE_DIM)
        
        for i, feature_name in enumerate(self._feature_names):
            if feature_name in features and features[feature_name] is not None:
                value = features[feature_name]
                
                if not np.isnan(value):
                    scaler = self.scalers.get(feature_name)
                    if scaler:
                        normalized = scaler.transform([[value]])[0, 0]
                        state[i] = normalized
                    else:
                        state[i] = value
            # If missing, leave as 0 (neutral)
        
        # Clip extreme values
        state = np.clip(state, self.clip_range[0], self.clip_range[1])
        
        return state
    
    def transform_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Transform a batch of features.
        
        Args:
            features_df: DataFrame with feature columns
            
        Returns:
            Array of shape (n_samples, 36)
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        n_samples = len(features_df)
        states = np.zeros((n_samples, STATE_DIM))
        
        for i, feature_name in enumerate(self._feature_names):
            if feature_name in features_df.columns:
                values = features_df[feature_name].values.reshape(-1, 1)
                
                # Handle NaN values
                nan_mask = np.isnan(values.flatten())
                
                scaler = self.scalers.get(feature_name)
                if scaler:
                    normalized = scaler.transform(values).flatten()
                    normalized[nan_mask] = 0.0  # Impute NaN as neutral
                    states[:, i] = normalized
        
        # Clip extreme values
        states = np.clip(states, self.clip_range[0], self.clip_range[1])
        
        return states
    
    def inverse_transform(self, state: np.ndarray) -> Dict[str, float]:
        """
        Convert normalized state back to original feature values.
        
        Args:
            state: Normalized state vector
            
        Returns:
            Dictionary of original feature values
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        features = {}
        
        for i, feature_name in enumerate(self._feature_names):
            scaler = self.scalers.get(feature_name)
            if scaler:
                value = scaler.inverse_transform([[state[i]]])[0, 0]
                features[feature_name] = value
        
        return features
    
    def save(self, path: str):
        """Save preprocessor to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'scalers': self.scalers,
                'fitted': self.fitted,
                'feature_names': self._feature_names,
                'clip_range': self.clip_range
            }, f)
        
        logger.info(f"Preprocessor saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'StatePreprocessor':
        """Load preprocessor from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        preprocessor = cls(clip_range=data['clip_range'])
        preprocessor.scalers = data['scalers']
        preprocessor.fitted = data['fitted']
        preprocessor._feature_names = data['feature_names']
        
        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self._feature_names.copy()


def create_default_features() -> Dict[str, float]:
    """
    Create a dictionary with default (neutral) feature values.
    
    Useful for initializing or when data is missing.
    """
    defaults = {}
    
    for feature_name, (min_val, max_val, _) in FEATURE_SPECS.items():
        # Use midpoint as default
        defaults[feature_name] = (min_val + max_val) / 2
    
    return defaults


def validate_features(features: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Validate feature values are within expected ranges.
    
    Args:
        features: Dictionary of feature values
        
    Returns:
        Tuple of (is_valid, list of warnings)
    """
    warnings = []
    
    for feature_name, value in features.items():
        if feature_name in FEATURE_SPECS:
            min_val, max_val, _ = FEATURE_SPECS[feature_name]
            
            if np.isnan(value):
                warnings.append(f"{feature_name}: NaN value")
            elif value < min_val - 0.1 or value > max_val + 0.1:
                warnings.append(f"{feature_name}: {value:.3f} outside range [{min_val}, {max_val}]")
    
    return len(warnings) == 0, warnings


def get_feature_info(feature_name: str) -> Dict[str, any]:
    """Get information about a specific feature."""
    if feature_name in FEATURE_SPECS:
        min_val, max_val, description = FEATURE_SPECS[feature_name]
        return {
            'name': feature_name,
            'min': min_val,
            'max': max_val,
            'description': description,
            'index': FEATURE_NAMES.index(feature_name)
        }
    return None
