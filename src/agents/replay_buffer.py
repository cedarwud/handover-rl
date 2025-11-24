#!/usr/bin/env python3
"""
Experience Replay Buffer for DQN

Stores and samples experience transitions for off-policy RL training.

Transition: (state, action, reward, next_state, done)

Features:
- Circular buffer with fixed capacity
- Uniform random sampling
- Batch sampling for training

SOURCE:
- Mnih et al. (2015) "Human-level control through deep reinforcement learning"
  Nature 518(7540): 529-533
- Lin (1992) "Self-improving reactive agents based on reinforcement learning"
"""

import numpy as np
import random
from typing import List, Tuple, Optional
from collections import deque


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.

    Stores transitions and samples random batches for training.

    Usage:
        buffer = ReplayBuffer(capacity=100000)
        buffer.push(state, action, reward, next_state, done)
        batch = buffer.sample(batch_size=32)
    """

    def __init__(self, capacity: int = 100000, seed: Optional[int] = None):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            seed: Random seed for reproducibility
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0

        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def push(self,
             state: np.ndarray,
             action: int,
             reward: float,
             next_state: np.ndarray,
             done: bool):
        """
        Add a transition to the buffer.

        Args:
            state: Current state (numpy array)
            action: Action taken (int)
            reward: Reward received (float)
            next_state: Next state (numpy array)
            done: Episode termination flag (bool)
        """
        # Store as tuple
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            states: Batch of states (batch_size, state_dim)
            actions: Batch of actions (batch_size,)
            rewards: Batch of rewards (batch_size,)
            next_states: Batch of next states (batch_size, state_dim)
            dones: Batch of done flags (batch_size,)
        """
        # Sample random indices
        batch = random.sample(self.buffer, batch_size)

        # Unzip transitions
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer has enough samples for training.

        Args:
            batch_size: Required batch size

        Returns:
            ready: True if buffer size >= batch_size
        """
        return len(self.buffer) >= batch_size

    def clear(self):
        """Clear all transitions from buffer."""
        self.buffer.clear()
        self.position = 0

    def save(self, filepath: str):
        """
        Save buffer to disk.

        Args:
            filepath: Path to save buffer (numpy format)
        """
        # Convert buffer to list
        buffer_list = list(self.buffer)

        # Save as numpy archive
        np.savez_compressed(
            filepath,
            buffer=buffer_list,
            capacity=self.capacity,
            size=len(self.buffer)
        )

    def load(self, filepath: str):
        """
        Load buffer from disk.

        Args:
            filepath: Path to load buffer from
        """
        # Load numpy archive
        data = np.load(filepath, allow_pickle=True)

        # Restore buffer
        self.capacity = int(data['capacity'])
        self.buffer = deque(data['buffer'].tolist(), maxlen=self.capacity)

    def get_statistics(self) -> dict:
        """
        Get buffer statistics.

        Returns:
            stats: Dictionary with buffer statistics
        """
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0,
                'avg_reward': 0.0,
                'action_distribution': [0, 0]
            }

        # Extract components
        _, actions, rewards, _, _ = zip(*self.buffer)
        actions = np.array(actions)
        rewards = np.array(rewards)

        # Calculate statistics
        stats = {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'avg_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'action_distribution': [
                int(np.sum(actions == 0)),  # Maintain
                int(np.sum(actions == 1))   # Handover
            ],
            'handover_rate': float(np.mean(actions))
        }

        return stats
