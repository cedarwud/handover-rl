#!/usr/bin/env python3
"""
DQN Agent for Multi-Satellite Handover

Academic Standard: Standard DQN (Nature 2015, Mnih et al.)
Based on: Graph RL paper (Aerospace 2024) + DQN best practices

Features:
- Q-Network for (K, 12) multi-satellite input
- Experience Replay Buffer
- Target Network with periodic updates
- ε-greedy exploration
- Online RL training (not offline dataset)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, Tuple, Optional
import logging

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

        logger.info(f"Q-Network initialized: {self.input_dim} → {hidden_dim} → {self.output_dim}")

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


class ReplayBuffer:
    """
    Experience Replay Buffer

    Stores agent's own exploration experiences for training
    Based on: DQN (Nature 2015) - NOT pre-labeled ground truth

    Academic Compliance:
    - Stores AGENT's experiences (online RL)
    - NOT pre-computed optimal actions
    - NOT offline dataset with labels
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer

        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)
        logger.info(f"ReplayBuffer initialized: capacity={capacity}")

    def push(self, state, action, reward, next_state, done):
        """
        Add experience to buffer

        Args:
            state: (K, 12) current state
            action: Integer action taken
            reward: Scalar reward received
            next_state: (K, 12) next state
            done: Boolean episode done flag
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample random batch from buffer

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of batched (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)

        # Unzip batch
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy arrays then tensors
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent for Multi-Satellite Handover

    Based on:
    - Standard DQN (Nature 2015, Mnih et al.)
    - Graph RL paper (Aerospace 2024) for multi-satellite application

    Key Features:
    - Online RL (agent explores environment)
    - Experience replay
    - Target network
    - ε-greedy exploration
    """

    def __init__(self, observation_space, action_space, config: Dict):
        """
        Initialize DQN agent

        Args:
            observation_space: Gym observation space (Box)
            action_space: Gym action space (Discrete)
            config: Configuration dictionary
        """
        # Extract dimensions from spaces
        self.obs_shape = observation_space.shape  # (K, 12)
        self.n_actions = action_space.n  # K+1
        self.max_visible_satellites = self.obs_shape[0]
        self.state_dim = self.obs_shape[1]

        # Get config parameters (no hardcoding)
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

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=self.buffer_capacity)

        # Training statistics
        self.training_steps = 0
        self.episode_count = 0
        self.loss_history = []

        logger.info(f"DQNAgent initialized")
        logger.info(f"  Observation space: {self.obs_shape}")
        logger.info(f"  Action space: {self.n_actions}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Gamma: {self.gamma}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Buffer capacity: {self.buffer_capacity}")

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using ε-greedy policy

        Args:
            state: (K, 12) observation
            training: If True, use ε-greedy; if False, greedy

        Returns:
            action: Integer action
        """
        # ε-greedy exploration
        if training and random.random() < self.epsilon:
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

    def train_step(self) -> Optional[float]:
        """
        Perform one training step

        Returns:
            loss: Training loss (None if not enough experiences)
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

        # Record loss
        loss_value = loss.item()
        self.loss_history.append(loss_value)

        return loss_value

    def update_epsilon(self):
        """Decay epsilon for exploration"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """Save agent to file"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episode_count': self.episode_count,
        }, filepath)
        logger.info(f"Agent saved to {filepath}")

    def load(self, filepath: str):
        """Load agent from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        self.episode_count = checkpoint['episode_count']
        logger.info(f"Agent loaded from {filepath}")
