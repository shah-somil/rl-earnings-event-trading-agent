"""
Experience Replay Buffer for DQN.

Implements a fixed-size buffer with uniform random sampling.
"""

import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional
import numpy as np

# Experience tuple
Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done'
])


class ReplayBuffer:
    """
    Fixed-size buffer for storing experience tuples.
    
    Implements uniform random sampling for training the DQN.
    """
    
    def __init__(self, capacity: int = 10000, seed: int = None):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            seed: Random seed for reproducibility
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
        """
        experience = Experience(
            state=np.array(state, dtype=np.float32),
            action=action,
            reward=float(reward),
            next_state=np.array(next_state, dtype=np.float32),
            done=done
        )
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        experiences = random.sample(self.buffer, batch_size)
        
        states = np.array([e.state for e in experiences], dtype=np.float32)
        actions = np.array([e.action for e in experiences], dtype=np.int64)
        rewards = np.array([e.reward for e in experiences], dtype=np.float32)
        next_states = np.array([e.next_state for e in experiences], dtype=np.float32)
        dones = np.array([e.done for e in experiences], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= min_size
    
    def clear(self):
        """Clear all experiences from buffer."""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    
    Samples experiences based on TD error priority.
    (Optional extension - can use basic ReplayBuffer for initial implementation)
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        seed: int = None
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent
            beta_increment: Beta increase per sample
            seed: Random seed
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        if seed is not None:
            np.random.seed(seed)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: float = None
    ):
        """Add experience with priority."""
        experience = Experience(
            state=np.array(state, dtype=np.float32),
            action=action,
            reward=float(reward),
            next_state=np.array(next_state, dtype=np.float32),
            done=done
        )
        
        # Default priority is max priority seen
        if priority is None:
            priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch with priorities."""
        if self.size < batch_size:
            batch_size = self.size
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Extract experiences
        experiences = [self.buffer[i] for i in indices]
        
        states = np.array([e.state for e in experiences], dtype=np.float32)
        actions = np.array([e.action for e in experiences], dtype=np.int64)
        rewards = np.array([e.reward for e in experiences], dtype=np.float32)
        next_states = np.array([e.next_state for e in experiences], dtype=np.float32)
        dones = np.array([e.done for e in experiences], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small constant for stability
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, min_size: int) -> bool:
        return self.size >= min_size
