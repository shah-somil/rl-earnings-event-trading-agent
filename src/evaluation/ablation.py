"""
Ablation Studies for EETA.

Runs experiments to prove each component adds value:
- Full System
- No Historical Agent
- No Sentiment Agent  
- No Thompson Sampling
- Random Actions
"""

import logging
from typing import Dict, Any, List, Callable, Optional
import numpy as np
import pandas as pd
from copy import deepcopy

from ..rl.actions import Actions

# DQNAgent imported only when needed
try:
    from ..rl.dqn import DQNAgent
except ImportError:
    DQNAgent = None
from ..rl.thompson import ThompsonSampler
from ..environment.trading_env import EarningsTradingEnv
from ..data.preprocessor import StatePreprocessor
from .metrics import calculate_all_metrics

logger = logging.getLogger(__name__)


class AblationExperiment:
    """Single ablation experiment configuration."""
    
    def __init__(
        self,
        name: str,
        description: str,
        modifier: Callable = None
    ):
        """
        Initialize experiment.
        
        Args:
            name: Experiment name
            description: What is disabled/changed
            modifier: Function to modify agent/environment
        """
        self.name = name
        self.description = description
        self.modifier = modifier


class AblationStudy:
    """
    Runs ablation experiments to validate component contributions.
    
    Experiments:
    1. Full System - All components enabled
    2. No Historical - Historical agent outputs zeroed
    3. No Sentiment - Sentiment agent outputs zeroed
    4. No Thompson - Fixed position sizing
    5. Random Actions - No DQN learning
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        preprocessor: StatePreprocessor,
        config: Dict[str, Any]
    ):
        """
        Initialize ablation study.
        
        Args:
            data: Test dataset
            preprocessor: Fitted preprocessor
            config: Training configuration
        """
        self.data = data
        self.preprocessor = preprocessor
        self.config = config
        
        # Define experiments
        self.experiments = self._define_experiments()
    
    def _define_experiments(self) -> List[AblationExperiment]:
        """Define ablation experiments."""
        return [
            AblationExperiment(
                name="full_system",
                description="Full system with all components"
            ),
            AblationExperiment(
                name="no_historical",
                description="Historical features zeroed",
                modifier=self._zero_historical_features
            ),
            AblationExperiment(
                name="no_sentiment",
                description="Sentiment features zeroed",
                modifier=self._zero_sentiment_features
            ),
            AblationExperiment(
                name="no_thompson",
                description="Fixed position sizing (2%)",
                modifier=self._fixed_position_size
            ),
            AblationExperiment(
                name="random_actions",
                description="Random action selection",
                modifier=self._random_actions
            )
        ]
    
    def run(
        self,
        trained_agent: DQNAgent,
        n_episodes: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all ablation experiments.
        
        Args:
            trained_agent: Pre-trained DQN agent
            n_episodes: Episodes per experiment
            
        Returns:
            Results dictionary
        """
        results = {}
        
        for experiment in self.experiments:
            logger.info(f"Running ablation: {experiment.name}")
            logger.info(f"  Description: {experiment.description}")
            
            # Run experiment
            exp_results = self._run_experiment(
                experiment, trained_agent, n_episodes
            )
            
            results[experiment.name] = {
                'description': experiment.description,
                **exp_results
            }
            
            logger.info(f"  Sharpe: {exp_results['sharpe_ratio']:.2f}")
            logger.info(f"  Return: {exp_results['total_return']:.2%}")
        
        # Calculate relative performance
        self._calculate_relative_performance(results)
        
        return results
    
    def _run_experiment(
        self,
        experiment: AblationExperiment,
        agent: DQNAgent,
        n_episodes: int
    ) -> Dict[str, Any]:
        """Run a single ablation experiment."""
        all_returns = []
        all_actions = {i: 0 for i in range(5)}
        
        for ep in range(n_episodes):
            # Create environment
            env = EarningsTradingEnv(
                data=self.data,
                preprocessor=self.preprocessor,
                shuffle=True
            )
            
            state = env.reset()
            episode_returns = []
            
            while not env.done:
                # Apply experiment modifier
                if experiment.modifier:
                    modified = experiment.modifier(state, agent)
                    if modified is not None:
                        if isinstance(modified, int):
                            # Modifier returned an action
                            action = modified
                            position_size = 0.02
                        elif isinstance(modified, tuple):
                            state, action, position_size = modified
                            if action is None:
                                action = agent.select_action(state, greedy=True)
                        else:
                            state = modified
                            action = agent.select_action(state, greedy=True)
                            position_size = 0.02
                    else:
                        action = agent.select_action(state, greedy=True)
                        position_size = 0.02
                else:
                    action = agent.select_action(state, greedy=True)
                    position_size = 0.02
                
                next_state, reward, done, info = env.step(action, position_size)
                
                if action != Actions.NO_TRADE:
                    pnl = info.get('pnl_pct', 0.0)
                    episode_returns.append(pnl)
                
                all_actions[action] += 1
                state = next_state
            
            all_returns.extend(episode_returns)
        
        # Calculate metrics
        metrics = calculate_all_metrics(all_returns)
        metrics['action_distribution'] = all_actions
        
        return metrics
    
    def _zero_historical_features(
        self,
        state: np.ndarray,
        agent: DQNAgent
    ) -> np.ndarray:
        """Zero out historical features (indices 0-11)."""
        modified_state = state.copy()
        modified_state[0:12] = 0.0
        return modified_state
    
    def _zero_sentiment_features(
        self,
        state: np.ndarray,
        agent: DQNAgent
    ) -> np.ndarray:
        """Zero out sentiment features (indices 12-19)."""
        modified_state = state.copy()
        modified_state[12:20] = 0.0
        return modified_state
    
    def _fixed_position_size(
        self,
        state: np.ndarray,
        agent: DQNAgent
    ) -> tuple:
        """Use fixed position size instead of Thompson Sampling."""
        action = agent.select_action(state, greedy=True)
        return (state, action, 0.02)  # Fixed 2% position
    
    def _random_actions(
        self,
        state: np.ndarray,
        agent: DQNAgent
    ) -> int:
        """Return random action instead of using DQN."""
        return np.random.randint(5)
    
    def _calculate_relative_performance(self, results: Dict[str, Dict[str, Any]]):
        """Calculate performance relative to full system."""
        if 'full_system' not in results:
            return
        
        full_sharpe = results['full_system'].get('sharpe_ratio', 0)
        full_return = results['full_system'].get('total_return', 0)
        
        for name, res in results.items():
            res['sharpe_delta'] = res.get('sharpe_ratio', 0) - full_sharpe
            res['return_delta'] = res.get('total_return', 0) - full_return
            
            if full_sharpe != 0:
                res['sharpe_pct_change'] = (res['sharpe_delta'] / abs(full_sharpe)) * 100
            else:
                res['sharpe_pct_change'] = 0
    
    def get_summary_table(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Generate summary table of ablation results."""
        rows = []
        
        for name, res in results.items():
            rows.append({
                'Experiment': name,
                'Description': res.get('description', ''),
                'Sharpe Ratio': res.get('sharpe_ratio', 0),
                'Sharpe Δ': res.get('sharpe_delta', 0),
                'Sharpe Δ%': res.get('sharpe_pct_change', 0),
                'Total Return': res.get('total_return', 0),
                'Win Rate': res.get('win_rate', 0),
                'N Trades': res.get('n_trades', 0)
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('Sharpe Ratio', ascending=False)
        
        return df
    
    def get_component_importance(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate importance of each component.
        
        Importance = how much Sharpe drops when component is removed.
        """
        if 'full_system' not in results:
            return {}
        
        full_sharpe = results['full_system'].get('sharpe_ratio', 0)
        
        importance = {}
        
        if 'no_historical' in results:
            importance['Historical Agent'] = full_sharpe - results['no_historical'].get('sharpe_ratio', 0)
        
        if 'no_sentiment' in results:
            importance['Sentiment Agent'] = full_sharpe - results['no_sentiment'].get('sharpe_ratio', 0)
        
        if 'no_thompson' in results:
            importance['Thompson Sampling'] = full_sharpe - results['no_thompson'].get('sharpe_ratio', 0)
        
        if 'random_actions' in results:
            importance['DQN Learning'] = full_sharpe - results['random_actions'].get('sharpe_ratio', 0)
        
        return importance


def run_ablation_study(
    agent: DQNAgent,
    test_data: pd.DataFrame,
    preprocessor: StatePreprocessor,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Convenience function to run ablation study.
    
    Args:
        agent: Trained DQN agent
        test_data: Test dataset
        preprocessor: Fitted preprocessor
        config: Optional configuration
        
    Returns:
        Ablation results
    """
    config = config or {}
    
    study = AblationStudy(test_data, preprocessor, config)
    results = study.run(agent, n_episodes=config.get('n_episodes', 10))
    
    # Generate summary
    summary = study.get_summary_table(results)
    importance = study.get_component_importance(results)
    
    return {
        'results': results,
        'summary': summary,
        'component_importance': importance
    }
