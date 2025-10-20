"""
RL Agents - Concrete Algorithm Implementations

This module provides concrete implementations of RL algorithms for
satellite handover optimization.

Available Agents:
    DQNAgent: Deep Q-Network with experience replay and target network

Network Architectures:
    DQNNetwork: Standard DQN architecture
    DuelingDQNNetwork: Dueling DQN architecture (optional)

Utilities:
    ReplayBuffer: Standard experience replay buffer
    PrioritizedReplayBuffer: Prioritized experience replay (optional)

Usage:
    from src.agents import DQNAgent

    config = {
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'use_double_dqn': True
    }

    agent = DQNAgent(state_dim=12, action_dim=2, config=config)
    action = agent.select_action(state)
    loss = agent.update()
"""

from .dqn_agent import DQNAgent
from .dqn_network import DQNNetwork, DuelingDQNNetwork
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

__all__ = [
    'DQNAgent',
    'DQNNetwork',
    'DuelingDQNNetwork',
    'ReplayBuffer',
    'PrioritizedReplayBuffer'
]

__version__ = "2.0.0"
