#!/usr/bin/env python3
"""
Experience Replay Buffer for DQN

Stores and samples experience transitions for off-policy RL training.

Transition: (state, action, reward, next_state, done)

Features:
- Circular buffer with fixed capacity
- Uniform random sampling
- Batch sampling for training
- Optional prioritized sampling (future extension)

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


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer (optional extension).

    Samples transitions based on TD-error priorities.

    SOURCE: Schaul et al. (2016) "Prioritized Experience Replay", ICLR
    """

    def __init__(self,
                 capacity: int = 100000,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001,
                 epsilon: float = 1e-6,
                 seed: Optional[int] = None):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0=uniform, 1=full prioritization)
            beta: Importance sampling exponent (0=no correction, 1=full correction)
            beta_increment: Increment beta per sample
            epsilon: Small constant to prevent zero priorities
            seed: Random seed
        """
        super().__init__(capacity, seed)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        # Priority storage (synchronized with buffer)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def push(self,
             state: np.ndarray,
             action: int,
             reward: float,
             next_state: np.ndarray,
             done: bool,
             priority: Optional[float] = None):
        """
        Add transition with priority.

        Args:
            state, action, reward, next_state, done: Transition components
            priority: Priority value (if None, use max priority)
        """
        # Add transition to buffer
        super().push(state, action, reward, next_state, done)

        # Add priority
        if priority is None:
            priority = self.max_priority

        self.priorities.append(priority)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample batch based on priorities.

        Returns:
            states, actions, rewards, next_states, dones, indices, weights
        """
        # Calculate sampling probabilities
        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices
        indices = np.random.choice(
            len(self.buffer),
            size=batch_size,
            replace=False,
            p=probabilities
        )

        # Get transitions
        transitions = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*transitions)

        # Calculate importance sampling weights
        # w_i = (N * P(i))^(-beta)
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize

        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Convert to numpy
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        weights = np.array(weights, dtype=np.float32)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for sampled transitions.

        Args:
            indices: Indices of transitions to update
            priorities: New priority values (e.g., TD errors)
        """
        for idx, priority in zip(indices, priorities):
            # Ensure priority > 0
            priority = max(priority, self.epsilon)

            # Update priority
            self.priorities[idx] = priority

            # Update max priority
            self.max_priority = max(self.max_priority, priority)


# Example usage
if __name__ == "__main__":
    print("Experience Replay Buffer")
    print("=" * 60)

    # Create buffer
    buffer = ReplayBuffer(capacity=1000, seed=42)
    print(f"✅ Buffer created: capacity={buffer.capacity}")

    # Add transitions
    print("\nAdding 100 random transitions...")
    for i in range(100):
        state = np.random.randn(12).astype(np.float32)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(12).astype(np.float32)
        done = (i % 20 == 19)  # Episode ends every 20 steps

        buffer.push(state, action, reward, next_state, done)

    print(f"✅ Buffer size: {len(buffer)}")

    # Check if ready
    batch_size = 32
    print(f"\nReady for training (batch_size={batch_size})? {buffer.is_ready(batch_size)}")

    # Sample batch
    if buffer.is_ready(batch_size):
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        print(f"\n📦 Sampled batch:")
        print(f"   States shape: {states.shape}")
        print(f"   Actions shape: {actions.shape}")
        print(f"   Rewards shape: {rewards.shape}")
        print(f"   Next states shape: {next_states.shape}")
        print(f"   Dones shape: {dones.shape}")

    # Get statistics
    stats = buffer.get_statistics()
    print(f"\n📊 Buffer statistics:")
    print(f"   Size: {stats['size']}/{stats['capacity']}")
    print(f"   Utilization: {stats['utilization']:.1%}")
    print(f"   Avg reward: {stats['avg_reward']:.3f}")
    print(f"   Action distribution: Maintain={stats['action_distribution'][0]}, "
          f"Handover={stats['action_distribution'][1]}")
    print(f"   Handover rate: {stats['handover_rate']:.3f}")

    # Test prioritized buffer
    print("\n" + "=" * 60)
    print("Prioritized Replay Buffer")
    print("=" * 60)

    pri_buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.6, beta=0.4)
    print(f"✅ Prioritized buffer created")

    # Add transitions
    for i in range(100):
        state = np.random.randn(12).astype(np.float32)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(12).astype(np.float32)
        done = (i % 20 == 19)

        # Random priority
        priority = np.random.rand()

        pri_buffer.push(state, action, reward, next_state, done, priority)

    # Sample with priorities
    if pri_buffer.is_ready(batch_size):
        result = pri_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones, indices, weights = result

        print(f"\n📦 Prioritized sample:")
        print(f"   Batch size: {len(states)}")
        print(f"   Weights shape: {weights.shape}")
        print(f"   Weights range: [{weights.min():.3f}, {weights.max():.3f}]")
        print(f"   Beta: {pri_buffer.beta:.3f}")

    print("\n" + "=" * 60)
    print("✅ Replay Buffer verified!")
