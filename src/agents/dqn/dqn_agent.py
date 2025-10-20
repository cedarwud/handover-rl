#!/usr/bin/env python3
"""
DQN Agent - Refactored to BaseAgent Interface

Deep Q-Network agent for satellite handover optimization.

Based on:
- Standard DQN (Nature 2015, Mnih et al.)
- Graph RL paper (Aerospace 2024) for multi-satellite application

Refactoring Changes:
- Inherits from BaseAgent (unified interface)
- Compatible with OffPolicyTrainer
- Uses Gymnasium API (observation_space, action_space)
- Implements on_episode_end() for epsilon decay
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Any
import logging

from ..base_agent import BaseAgent

logger = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """
    Q-Network for multi-satellite state

    Input: (batch, K, 12) - K satellites × 12 features
    Output: (batch, K+1) - Q-values for K+1 actions

    Architecture:
    - Flatten multi-satellite input
    - Fully connected layers
    - Output Q-values for each action
    """

    def __init__(self, max_visible_satellites: int = 10, state_dim: int = 12,
                 hidden_dim: int = 128):
        """
        Initialize Q-Network

        Args:
            max_visible_satellites: K (max satellites in observation)
            state_dim: Feature dimensions per satellite (12)
            hidden_dim: Hidden layer size
        """
        super(QNetwork, self).__init__()

        self.max_visible_satellites = max_visible_satellites
        self.state_dim = state_dim
        self.input_dim = max_visible_satellites * state_dim  # K × 12 = 120
        self.output_dim = max_visible_satellites + 1  # K+1 actions

        # Network layers
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.output_dim)

        # Activation
        self.relu = nn.ReLU()

        logger.debug(f"Q-Network: {self.input_dim} → {hidden_dim} → {self.output_dim}")

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (batch, K, 12) multi-satellite state

        Returns:
            q_values: (batch, K+1) Q-values for each action
        """
        # Flatten multi-satellite input: (batch, K, 12) → (batch, K*12)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        # Forward through network
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values


class DQNAgent(BaseAgent):
    """
    DQN Agent implementing BaseAgent interface

    Key Features:
    - Online RL (agent explores environment)
    - Experience replay (from ReplayBuffer)
    - Target network (periodic sync)
    - ε-greedy exploration

    BaseAgent Implementation:
    - select_action(): ε-greedy policy
    - update(): DQN loss with target network
    - save/load(): PyTorch checkpoint format
    - on_episode_end(): Epsilon decay callback
    """

    def __init__(self, observation_space, action_space, config: Dict):
        """
        Initialize DQN agent

        Args:
            observation_space: Gym observation space (Box)
            action_space: Gym action space (Discrete)
            config: Configuration dictionary

        Config Parameters:
            agent:
                learning_rate: Adam learning rate (default: 1e-4)
                gamma: Discount factor (default: 0.99)
                batch_size: Training batch size (default: 64)
                buffer_capacity: Replay buffer size (default: 10000)
                target_update_freq: Target network update frequency (default: 100)
                hidden_dim: Q-network hidden dimension (default: 128)
                epsilon_start: Initial epsilon (default: 1.0)
                epsilon_end: Final epsilon (default: 0.05)
                epsilon_decay: Epsilon decay rate (default: 0.995)
        """
        super().__init__()

        # Extract dimensions from Gymnasium spaces
        self.obs_shape = observation_space.shape  # (K, 12)
        self.n_actions = action_space.n  # K+1
        self.max_visible_satellites = self.obs_shape[0]
        self.state_dim = self.obs_shape[1]

        # Get config parameters
        agent_config = config.get('agent', {})
        self.learning_rate = agent_config.get('learning_rate', 1e-4)
        self.gamma = agent_config.get('gamma', 0.99)
        self.batch_size = agent_config.get('batch_size', 64)
        self.buffer_capacity = agent_config.get('buffer_capacity', 10000)
        self.target_update_freq = agent_config.get('target_update_freq', 100)
        self.hidden_dim = agent_config.get('hidden_dim', 128)

        # Exploration parameters
        self.epsilon_start = agent_config.get('epsilon_start', 1.0)
        self.epsilon_end = agent_config.get('epsilon_end', 0.05)
        self.epsilon_decay = agent_config.get('epsilon_decay', 0.995)
        self.epsilon = self.epsilon_start

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Networks
        self.q_network = QNetwork(
            max_visible_satellites=self.max_visible_satellites,
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)

        self.target_network = QNetwork(
            max_visible_satellites=self.max_visible_satellites,
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)

        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Loss function
        self.criterion = nn.MSELoss()

        # Replay buffer (import here to avoid circular dependency)
        from ..replay_buffer import ReplayBuffer
        self.replay_buffer = ReplayBuffer(capacity=self.buffer_capacity)

        # Training statistics
        self.training_steps = 0
        self.episode_count = 0

        logger.info(f"DQNAgent initialized")
        logger.info(f"  Observation space: {self.obs_shape}")
        logger.info(f"  Action space: {self.n_actions}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Gamma: {self.gamma}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Buffer capacity: {self.buffer_capacity}")

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action using ε-greedy policy

        Args:
            state: (K, 12) observation
            deterministic: If True, use greedy policy (no exploration)
                          If False, use ε-greedy policy

        Returns:
            action: Integer action index
        """
        import random

        # ε-greedy exploration
        if not deterministic and random.random() < self.epsilon:
            # Explore: random action
            action = random.randrange(self.n_actions)
        else:
            # Exploit: greedy action based on Q-values
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax(dim=1).item()

        return action

    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer

        Args:
            state: (K, 12) current state
            action: Action taken
            reward: Reward received
            next_state: (K, 12) next state
            done: Episode done flag
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self, *args, **kwargs) -> Optional[float]:
        """
        Perform one DQN training step

        This method samples from the replay buffer and updates the Q-network
        using the DQN loss with target network.

        Returns:
            loss: Training loss (float)
            OR None if replay buffer doesn't have enough experiences

        DQN Update Formula:
            Loss = MSE(Q(s,a), r + γ * max_a' Q_target(s',a'))

        Target Network:
            Updated every `target_update_freq` steps
        """
        # Need enough experiences before training
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values: Q(s, a)
        current_q_values = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q-values: max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(dim=1)[0]

            # Target Q-values: r + γ * max_a' Q_target(s', a')
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update target network periodically
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            logger.debug(f"Target network updated at step {self.training_steps}")

        return loss.item()

    def save(self, path: str) -> None:
        """Save agent to file"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episode_count': self.episode_count,
        }, path)
        logger.info(f"DQNAgent saved to {path}")

    def load(self, path: str) -> None:
        """Load agent from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        self.episode_count = checkpoint['episode_count']
        logger.info(f"DQNAgent loaded from {path}")

    # ========== BaseAgent Callbacks ==========

    def on_episode_end(self, episode_reward: float, episode_info: Dict[str, Any]) -> None:
        """
        Called at the end of each episode

        Handles epsilon decay for exploration schedule.

        Args:
            episode_reward: Total episode reward
            episode_info: Episode information dict
        """
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episode_count += 1

    def get_config(self) -> Dict[str, Any]:
        """
        Return agent configuration

        Returns:
            config: Dictionary of hyperparameters
        """
        return {
            'algorithm': 'DQN',
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'buffer_capacity': self.buffer_capacity,
            'target_update_freq': self.target_update_freq,
            'hidden_dim': self.hidden_dim,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'training_steps': self.training_steps,
            'episode_count': self.episode_count,
            'buffer_size': len(self.replay_buffer),
        }
