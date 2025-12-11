"""
Deep Q-Network (DQN) for Position Selection.

Implements the DQN algorithm with:
- Target network for stable training
- Experience replay
- Epsilon-greedy exploration with decay
"""

import logging
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path

from src.rl.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """
    Neural network for Q-value approximation.
    
    Architecture: 43 → 128 → 64 → 5 (with dropout)
    """
    
    def __init__(
        self,
        state_dim: int = 43,
        action_dim: int = 5,
        hidden_dims: List[int] = None,
        dropout: float = 0.2
    ):
        """
        Initialize Q-network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation - raw Q-values)
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Q-values of shape (batch_size, action_dim)
        """
        return self.network(state)


class DQNAgent:
    """
    DQN Agent for position selection.
    
    Learns to select optimal trading positions using:
    - Deep Q-Network for Q-value approximation
    - Target network for stable training
    - Experience replay for sample efficiency
    - Epsilon-greedy exploration
    """
    
    def __init__(
        self,
        state_dim: int = 43,
        action_dim: int = 5,
        hidden_dims: List[int] = None,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        tau: float = 0.005,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        buffer_size: int = 10000,
        min_buffer_size: int = 1000,
        target_update_freq: int = 100,
        device: str = None,
        seed: int = None
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer sizes
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            tau: Soft update parameter for target network
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            batch_size: Training batch size
            buffer_size: Replay buffer capacity
            min_buffer_size: Minimum samples before training
            target_update_freq: Steps between target network updates
            device: Device for computation ('cpu' or 'cuda')
            seed: Random seed
        """
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Store hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.target_update_freq = target_update_freq
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Create networks
        self.q_network = QNetwork(
            state_dim, action_dim, hidden_dims
        ).to(self.device)
        
        self.target_network = QNetwork(
            state_dim, action_dim, hidden_dims
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, seed)
        
        # Training tracking
        self.training_steps = 0
        self.episodes = 0
        self.losses = []
    
    def select_action(
        self,
        state: np.ndarray,
        epsilon: float = None,
        greedy: bool = False
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Override exploration rate
            greedy: If True, always select best action
            
        Returns:
            Selected action index
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if greedy:
            epsilon = 0.0
        
        # Exploration
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        
        # Exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions.
        
        Args:
            state: Current state
            
        Returns:
            Array of Q-values for each action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy().flatten()
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        # Check if enough samples
        if not self.replay_buffer.is_ready(self.min_buffer_size):
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update target network
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self._soft_update_target()
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def _soft_update_target(self):
        """Soft update target network weights."""
        for target_param, local_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def end_episode(self):
        """Called at end of episode."""
        self.episodes += 1
        self.decay_epsilon()
    
    def save(self, path: str):
        """
        Save agent state to file.
        
        Args:
            path: Path to save file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episodes': self.episodes,
            'losses': self.losses[-1000:],  # Save recent losses
        }, path)
        
        logger.info(f"Agent saved to {path}")
    
    def load(self, path: str):
        """
        Load agent state from file.
        
        Args:
            path: Path to save file
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        self.episodes = checkpoint['episodes']
        self.losses = checkpoint.get('losses', [])
        
        logger.info(f"Agent loaded from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'episodes': self.episodes,
            'training_steps': self.training_steps,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0.0,
        }
    
    def set_training_mode(self, training: bool = True):
        """Set network training mode."""
        if training:
            self.q_network.train()
        else:
            self.q_network.eval()
            self.target_network.eval()


# Action space constants
class Actions:
    """Trading action definitions."""
    NO_TRADE = 0
    LONG_STOCK = 1
    SHORT_STOCK = 2
    LONG_VOL = 3      # Simulated straddle
    SHORT_VOL = 4     # Simulated iron condor
    
    NAMES = ['NO_TRADE', 'LONG', 'SHORT', 'LONG_VOL', 'SHORT_VOL']
    
    @classmethod
    def name(cls, action: int) -> str:
        """Get action name."""
        return cls.NAMES[action] if 0 <= action < len(cls.NAMES) else 'UNKNOWN'
