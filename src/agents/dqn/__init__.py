"""
DQN Module - Deep Q-Network Implementation

Standard DQN implementation for satellite handover optimization.

Based on:
- DQN (Nature 2015, Mnih et al.)
- Graph RL paper (Aerospace 2024)

Components:
    DQNAgent: Main agent class implementing BaseAgent interface
    QNetwork: Q-network architecture (internal use)

Usage:
    from src.agents.dqn import DQNAgent

    agent = DQNAgent(obs_space, action_space, config)
    action = agent.select_action(state, deterministic=False)
    loss = agent.update()
"""

from .dqn_agent import DQNAgent, QNetwork

__all__ = [
    'DQNAgent',
    'QNetwork',
]
