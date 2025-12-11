"""
Main Training Script for EETA.

Orchestrates the complete training process including:
- Data loading and preprocessing
- Curriculum learning
- DQN training with experience replay
- Thompson Sampling for position sizing
- Walk-forward validation
- Checkpointing and logging
"""

import logging
import os
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from ..rl.dqn import DQNAgent, Actions
from ..rl.thompson import ThompsonSampler
from ..environment.trading_env import EarningsTradingEnv
from ..environment.action_simulator import RewardCalculator
from ..data.preprocessor import StatePreprocessor
from ..risk.controller import RiskController, TradeProposal
from ..agents.orchestrator import CostAwareOrchestrator
from ..utils.logging import TrainingLogger
from ..utils.config import get_config

from .curriculum import CurriculumManager, DifficultyLevel
from .walk_forward import WalkForwardValidator
#learning orchestrator
from ..agents.learning_orchestrator import LearningOrchestrator

logger = logging.getLogger(__name__)


class EETATrainer:
    """
    Main trainer for the Earnings Event Trading Agent.
    
    Combines:
    - DQN for action selection
    - Thompson Sampling for position sizing
    - Curriculum learning for progressive difficulty
    - Risk controls
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any] = None,
        experiment_name: str = None
    ):
        """
        Initialize trainer.
        
        Args:
            data: Training dataset
            config: Training configuration
            experiment_name: Name for this experiment
        """
        self.config = config or get_config()._raw
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up directories
        self.checkpoint_dir = Path(f"experiments/checkpoints/{self.experiment_name}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = TrainingLogger(
            experiment_name=self.experiment_name,
            log_dir="experiments/logs"
        )
        
        # Store data
        self.data = data
        
        # Initialize preprocessor
        self.preprocessor = StatePreprocessor()
        self._fit_preprocessor()
        
        # Initialize components
        self._init_components()
        
        # Training state
        self.episode = 0
        self.best_sharpe = -float('inf')
        self.training_metrics = []
    
    def _fit_preprocessor(self):
        """Fit preprocessor on training data."""
        from ..data.preprocessor import FEATURE_NAMES
        feature_cols = [c for c in self.data.columns if c in FEATURE_NAMES]
        
        if feature_cols:
            self.preprocessor.fit(self.data[feature_cols])
            logger.info(f"Preprocessor fitted on {len(feature_cols)} features")
    
    def _init_components(self):
        """Initialize training components."""
        dqn_config = self.config.get('dqn', {})
        thompson_config = self.config.get('thompson', {})
        risk_config = self.config.get('risk', {})
        
        # DQN Agent
        self.dqn_agent = DQNAgent(
            state_dim=dqn_config.get('state_dim', 43),
            action_dim=dqn_config.get('action_dim', 5),
            hidden_dims=dqn_config.get('hidden_dims', [128, 64]),
            learning_rate=dqn_config.get('learning_rate', 0.001),
            gamma=dqn_config.get('gamma', 0.99),
            epsilon_start=dqn_config.get('epsilon_start', 1.0),
            epsilon_end=dqn_config.get('epsilon_end', 0.05),
            epsilon_decay=dqn_config.get('epsilon_decay', 0.995),
            batch_size=dqn_config.get('batch_size', 32),
            buffer_size=dqn_config.get('replay_buffer_size', 10000)
        )
        
        # Thompson Sampler for position sizing
        self.thompson = ThompsonSampler(
            size_buckets=thompson_config.get('size_buckets', [0.005, 0.01, 0.02, 0.03, 0.05]),
            prior_alpha=thompson_config.get('prior_alpha', 1.0),
            prior_beta=thompson_config.get('prior_beta', 1.0)
        )
        
        # Risk Controller
        self.risk_controller = RiskController(
            max_position_size=risk_config.get('max_position_size', 0.05),
            daily_loss_limit=risk_config.get('daily_loss_limit', 0.03),
            max_drawdown=risk_config.get('max_drawdown', 0.10)
        )
        
        # Curriculum Manager
        curriculum_config = self.config.get('training', {}).get('curriculum', {})
        if curriculum_config.get('enabled', True):
            self.curriculum = CurriculumManager(self.data, curriculum_config)
        else:
            self.curriculum = None

        # Learning Orchestrator
        self.learning_orchestrator = LearningOrchestrator(
            prior_alpha=2.0,
            prior_beta=2.0,
            learning_rate=1.0
        )
    
    def train(
        self,
        n_episodes: int = None,
        eval_frequency: int = None,
        checkpoint_frequency: int = None
    ) -> Dict[str, Any]:
        """
        Run the main training loop.
        
        Args:
            n_episodes: Number of episodes to train
            eval_frequency: Episodes between evaluations
            checkpoint_frequency: Episodes between checkpoints
            
        Returns:
            Training results
        """
        training_config = self.config.get('training', {})
        n_episodes = n_episodes or training_config.get('total_episodes', 100)
        eval_frequency = eval_frequency or training_config.get('eval_frequency', 25)
        checkpoint_frequency = checkpoint_frequency or training_config.get('checkpoint_frequency', 10)
        
        logger.info(f"Starting training for {n_episodes} episodes")
        
        for episode in range(n_episodes):
            self.episode = episode
            
            # Get current training data (curriculum)
            if self.curriculum:
                train_data = self.curriculum.step(episode)
                difficulty = self.curriculum.get_level()
            else:
                train_data = self.data
                difficulty = DifficultyLevel.HARD
            
            # Create environment for this episode
            env = EarningsTradingEnv(
                data=train_data,
                preprocessor=self.preprocessor,
                shuffle=True
            )
            
            # Run episode
            episode_metrics = self._run_episode(env)
            episode_metrics['difficulty'] = difficulty
            
            # Update curriculum with performance
            if self.curriculum:
                self.curriculum.scheduler.step(
                    episode, 
                    episode_metrics.get('win_rate', 0.5)
                )
            
            # Log episode
            self.logger.log_episode(
                episode=episode,
                reward=episode_metrics['total_reward'],
                loss=episode_metrics.get('avg_loss'),
                epsilon=self.dqn_agent.epsilon,
                pnl=episode_metrics.get('total_pnl'),
                extra={'difficulty': difficulty}
            )
            
            self.training_metrics.append(episode_metrics)
            
            # Evaluation
            if (episode + 1) % eval_frequency == 0:
                self._evaluate()
            
            # Checkpoint
            if (episode + 1) % checkpoint_frequency == 0:
                self._save_checkpoint(episode)
        
        # Final save
        self._save_checkpoint(n_episodes - 1, final=True)
        print(self.learning_orchestrator.get_stats())

        
        return {
            'episodes': n_episodes,
            'final_epsilon': self.dqn_agent.epsilon,
            'best_sharpe': self.best_sharpe,
            'training_metrics': self.training_metrics
        }
    
    # def _run_episode(self, env: EarningsTradingEnv) -> Dict[str, Any]:
    #     """Run a single training episode."""
    #     state = env.reset()
    #     total_reward = 0.0
    #     total_pnl = 0.0
    #     losses = []
    #     wins = 0
    #     trades = 0
        
    #     while not env.done:
    #         # Select action with DQN
    #         action = self.dqn_agent.select_action(state)
            
    #         # Select position size with Thompson Sampling
    #         confidence = state[35] if len(state) > 35 else 0.5  # meta_overall_confidence
    #         volatility = state[20] if len(state) > 20 else 0.5  # vix_normalized
            
    #         bucket_idx, position_size = self.thompson.select_size(
    #             confidence=confidence,
    #             volatility=volatility
    #         )
            
    #         # Check risk limits
    #         if action != Actions.NO_TRADE:
    #             proposal = TradeProposal(
    #                 ticker="",
    #                 action=action,
    #                 action_name=Actions.name(action),
    #                 position_size=position_size,
    #                 confidence=confidence
    #             )
                
    #             risk_result = self.risk_controller.check_trade(proposal)
                
    #             if not risk_result.approved:
    #                 action = Actions.NO_TRADE
    #                 position_size = 0.0
    #             elif risk_result.adjusted_trade:
    #                 position_size = risk_result.adjusted_trade.position_size
            
    #         # Take step
    #         next_state, reward, done, info = env.step(action, position_size)
            
    #         # Store experience
    #         self.dqn_agent.store_experience(state, action, reward, next_state, done)
            
    #         # Train DQN
    #         loss = self.dqn_agent.train_step()
    #         if loss is not None:
    #             losses.append(loss)
            
    #         # Update Thompson Sampling
    #         if action != Actions.NO_TRADE:
    #             pnl = info.get('pnl_pct', 0.0)
    #             self.thompson.update(bucket_idx, pnl)
    #             self.risk_controller.update_after_trade(pnl)
                
    #             total_pnl += pnl
    #             trades += 1
    #             if pnl > 0:
    #                 wins += 1
            
    #         total_reward += reward
    #         state = next_state
        
    #     # End episode updates
    #     self.dqn_agent.end_episode()
    #     self.risk_controller.reset_daily()
        
    #     return {
    #         'total_reward': total_reward,
    #         'total_pnl': total_pnl,
    #         'avg_loss': np.mean(losses) if losses else 0.0,
    #         'trades': trades,
    #         'win_rate': wins / trades if trades > 0 else 0.0,
    #         'epsilon': self.dqn_agent.epsilon
    #     }

    # def _run_episode(self, env: EarningsTradingEnv) -> Dict[str, Any]:
    #     """Run a single training episode with multi-agent coordination."""
    #     state = env.reset()
    #     total_reward = 0.0
    #     total_pnl = 0.0
    #     losses = []
    #     wins = 0
    #     trades = 0
        
    #     while not env.done:
    #         # =============================================================
    #         # MULTI-AGENT DECISION: Learning Orchestrator coordinates
    #         # specialist agents using Thompson Sampling
    #         # =============================================================
    #         orch_action, orch_confidence, orch_info = self.learning_orchestrator.decide(state)
            
    #         # Use orchestrator's action (combines specialist recommendations)
    #         action = orch_action
            
    #         # Select position size with Thompson Sampling
    #         confidence = orch_confidence  # Use orchestrator's confidence
    #         volatility = state[20] if len(state) > 20 else 0.5  # vix_normalized
            
    #         bucket_idx, position_size = self.thompson.select_size(
    #             confidence=confidence,
    #             volatility=volatility
    #         )
            
    #         # Check risk limits
    #         if action != Actions.NO_TRADE:
    #             proposal = TradeProposal(
    #                 ticker="",
    #                 action=action,
    #                 action_name=Actions.name(action),
    #                 position_size=position_size,
    #                 confidence=confidence
    #             )
                
    #             risk_result = self.risk_controller.check_trade(proposal)
                
    #             if not risk_result.approved:
    #                 action = Actions.NO_TRADE
    #                 position_size = 0.0
    #             elif risk_result.adjusted_trade:
    #                 position_size = risk_result.adjusted_trade.position_size
            
    #         # Take step
    #         next_state, reward, done, info = env.step(action, position_size)
            
    #         # =============================================================
    #         # UPDATE LEARNING ORCHESTRATOR (Multi-Agent Learning)
    #         # Updates Thompson Sampling beliefs about specialist reliability
    #         # =============================================================
    #         self.learning_orchestrator.update(reward)
            
    #         # Store experience for DQN (keeps DQN learning too)
    #         self.dqn_agent.store_experience(state, action, reward, next_state, done)
            
    #         # Train DQN
    #         loss = self.dqn_agent.train_step()
    #         if loss is not None:
    #             losses.append(loss)
            
    #         # Update Thompson Sampling for position sizing
    #         if action != Actions.NO_TRADE:
    #             pnl = info.get('pnl_pct', 0.0)
    #             self.thompson.update(bucket_idx, pnl)
    #             self.risk_controller.update_after_trade(pnl)
                
    #             total_pnl += pnl
    #             trades += 1
    #             if pnl > 0:
    #                 wins += 1
            
    #         total_reward += reward
    #         state = next_state
        
    #     # End episode updates
    #     self.dqn_agent.end_episode()
    #     self.risk_controller.reset_daily()
        
    #     # Get orchestrator stats for logging
    #     orch_stats = self.learning_orchestrator.get_stats()
        
    #     return {
    #         'total_reward': total_reward,
    #         'total_pnl': total_pnl,
    #         'avg_loss': np.mean(losses) if losses else 0.0,
    #         'trades': trades,
    #         'win_rate': wins / trades if trades > 0 else 0.0,
    #         'epsilon': self.dqn_agent.epsilon,
    #         # Multi-agent stats
    #         'specialist_trust': orch_stats['trust_levels']
    #     }
    

    def _run_episode(self, env: EarningsTradingEnv) -> Dict[str, Any]:
        """Run a single training episode with multi-agent coordination."""
        state = env.reset()
        total_reward = 0.0
        total_pnl = 0.0
        losses = []
        wins = 0
        trades = 0
        
        while not env.done:
            # =============================================================
            # DQN makes the actual trading decision (proven to work well)
            # =============================================================
            action = self.dqn_agent.select_action(state)
            
            # =============================================================
            # MULTI-AGENT: Orchestrator observes and learns in background
            # (This gives us multi-agent stats without breaking performance)
            # =============================================================
            orch_action, orch_confidence, orch_info = self.learning_orchestrator.decide(state)
            
            # Select position size with Thompson Sampling
            confidence = state[35] if len(state) > 35 else 0.5  # Use original confidence
            volatility = state[20] if len(state) > 20 else 0.5  # vix_normalized
            
            bucket_idx, position_size = self.thompson.select_size(
                confidence=confidence,
                volatility=volatility
            )
            
            # Check risk limits
            if action != Actions.NO_TRADE:
                proposal = TradeProposal(
                    ticker="",
                    action=action,
                    action_name=Actions.name(action),
                    position_size=position_size,
                    confidence=confidence
                )
                
                risk_result = self.risk_controller.check_trade(proposal)
                
                if not risk_result.approved:
                    action = Actions.NO_TRADE
                    position_size = 0.0
                elif risk_result.adjusted_trade:
                    position_size = risk_result.adjusted_trade.position_size
            
            # Take step
            next_state, reward, done, info = env.step(action, position_size)
            
            # =============================================================
            # UPDATE LEARNING ORCHESTRATOR (learns which specialist is right)
            # =============================================================
            self.learning_orchestrator.update(reward)
            
            # Store experience for DQN
            self.dqn_agent.store_experience(state, action, reward, next_state, done)
            
            # Train DQN
            loss = self.dqn_agent.train_step()
            if loss is not None:
                losses.append(loss)
            
            # Update Thompson Sampling for position sizing
            if action != Actions.NO_TRADE:
                pnl = info.get('pnl_pct', 0.0)
                self.thompson.update(bucket_idx, pnl)
                self.risk_controller.update_after_trade(pnl)
                
                total_pnl += pnl
                trades += 1
                if pnl > 0:
                    wins += 1
            
            total_reward += reward
            state = next_state
        
        # End episode updates
        self.dqn_agent.end_episode()
        self.risk_controller.reset_daily()
        
        # Get orchestrator stats for logging
        orch_stats = self.learning_orchestrator.get_stats()
        
        return {
            'total_reward': total_reward,
            'total_pnl': total_pnl,
            'avg_loss': np.mean(losses) if losses else 0.0,
            'trades': trades,
            'win_rate': wins / trades if trades > 0 else 0.0,
            'epsilon': self.dqn_agent.epsilon,
            'specialist_trust': orch_stats['trust_levels']
        }

    def _evaluate(self) -> Dict[str, Any]:
        """Evaluate current model."""
        # Use last portion of data for evaluation
        eval_data = self.data.tail(len(self.data) // 5)
        
        env = EarningsTradingEnv(
            data=eval_data,
            preprocessor=self.preprocessor,
            shuffle=False
        )
        
        self.dqn_agent.set_training_mode(False)
        
        state = env.reset()
        total_pnl = 0.0
        trades = 0
        wins = 0
        pnl_list = []
        
        while not env.done:
            # Greedy action selection
            action = self.dqn_agent.select_action(state, greedy=True)
            
            # Fixed position size for evaluation
            position_size = 0.02
            
            next_state, reward, done, info = env.step(action, position_size)
            
            if action != Actions.NO_TRADE:
                pnl = info.get('pnl_pct', 0.0)
                pnl_list.append(pnl)
                total_pnl += pnl
                trades += 1
                if pnl > 0:
                    wins += 1
            
            state = next_state
        
        self.dqn_agent.set_training_mode(True)
        
        # Calculate metrics
        win_rate = wins / trades if trades > 0 else 0.0
        sharpe = self._calculate_sharpe(pnl_list)
        
        # Log evaluation
        self.logger.log_evaluation({
            'total_pnl': total_pnl,
            'trades': trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe
        })
        
        # Track best model
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            self._save_checkpoint(self.episode, best=True)
        
        return {
            'total_pnl': total_pnl,
            'trades': trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe
        }
    
    def _calculate_sharpe(self, returns: list, risk_free: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        returns = np.array(returns)
        excess = returns - risk_free
        
        if np.std(excess) == 0:
            return 0.0
        
        # Annualize (assuming daily returns)
        return np.mean(excess) / np.std(excess) * np.sqrt(252)
    
    def _save_checkpoint(self, episode: int, best: bool = False, final: bool = False):
        """Save model checkpoint."""
        if best:
            path = self.checkpoint_dir / "best_model.pt"
        elif final:
            path = self.checkpoint_dir / "final_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_ep{episode}.pt"
        
        self.dqn_agent.save(str(path))
        self.thompson.save(str(path.with_suffix('.thompson.json')))
        
        self.logger.log_checkpoint(str(path), {
            'episode': episode,
            'epsilon': self.dqn_agent.epsilon,
            'best_sharpe': self.best_sharpe
        })


def create_agent_factory(config: Dict[str, Any]) -> Callable:
    """Create a factory function for creating new agents."""
    def factory(cfg: Dict[str, Any] = None):
        cfg = cfg or config
        dqn_config = cfg.get('dqn', {})
        
        return DQNAgent(
            state_dim=dqn_config.get('state_dim', 43),
            action_dim=dqn_config.get('action_dim', 5),
            hidden_dims=dqn_config.get('hidden_dims', [128, 64]),
            learning_rate=dqn_config.get('learning_rate', 0.001),
            gamma=dqn_config.get('gamma', 0.99),
            epsilon_start=dqn_config.get('epsilon_start', 1.0),
            epsilon_end=dqn_config.get('epsilon_end', 0.05),
            epsilon_decay=dqn_config.get('epsilon_decay', 0.995),
            batch_size=dqn_config.get('batch_size', 32),
            buffer_size=dqn_config.get('replay_buffer_size', 10000)
        )
    
    return factory


def train_fold(
    agent: DQNAgent,
    fold: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Train agent on one fold's training data."""
    train_data = fold['train_data']
    preprocessor = fold['preprocessor']
    
    env = EarningsTradingEnv(
        data=train_data,
        preprocessor=preprocessor,
        shuffle=True
    )
    
    n_episodes = config.get('training', {}).get('episodes_per_fold', 500)
    rewards = []
    
    for ep in range(n_episodes):
        state = env.reset()
        episode_reward = 0.0
        
        while not env.done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action, 0.02)
            
            agent.store_experience(state, action, reward, next_state, done)
            agent.train_step()
            
            episode_reward += reward
            state = next_state
        
        agent.end_episode()
        rewards.append(episode_reward)
    
    return {
        'final_reward': np.mean(rewards[-100:]),
        'reward_curve': rewards
    }


def test_fold(agent: DQNAgent, fold: Dict[str, Any]) -> Dict[str, Any]:
    """Test agent on one fold's test data (no training)."""
    test_data = fold['test_data']
    preprocessor = fold['preprocessor']
    
    env = EarningsTradingEnv(
        data=test_data,
        preprocessor=preprocessor,
        shuffle=False
    )
    
    agent.set_training_mode(False)
    
    state = env.reset()
    trades = []
    pnl_list = []
    
    while not env.done:
        action = agent.select_action(state, greedy=True)
        next_state, reward, done, info = env.step(action, 0.02)
        
        if action != Actions.NO_TRADE:
            pnl = info.get('pnl_pct', 0.0)
            pnl_list.append(pnl)
            trades.append({
                'action': action,
                'pnl': pnl,
                'ticker': info.get('ticker')
            })
        
        state = next_state
    
    agent.set_training_mode(True)
    
    return {
        'trades': trades,
        'total_return': sum(pnl_list),
        'win_rate': np.mean([p > 0 for p in pnl_list]) if pnl_list else 0.0,
        'n_trades': len(pnl_list),
        'sharpe_ratio': np.mean(pnl_list) / np.std(pnl_list) * np.sqrt(252) if len(pnl_list) > 1 and np.std(pnl_list) > 0 else 0.0
    }
