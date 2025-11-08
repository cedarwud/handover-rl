"""
RL Agents - Modular Framework for Satellite Handover

This module provides the base agent interface and concrete implementations
of RL algorithms for satellite handover optimization.

Architecture:
    Phase 1 (Refactoring): BaseAgent interface + DQN refactored
    Phase 2 (Baselines): Rule-based comparison methods

Base Classes:
    BaseAgent: Abstract interface for all RL agents

Agents (will be added during refactoring):
    DQNAgent: Deep Q-Network (to be refactored in Task 1.3)

Network Architectures:
    DQNNetwork: Standard DQN architecture
    DuelingDQNNetwork: Dueling DQN architecture (optional)

Utilities:
    ReplayBuffer: Standard experience replay buffer
    PrioritizedReplayBuffer: Prioritized experience replay (optional)

Usage (after refactoring):
    from src.agents import BaseAgent, DQNAgent

    agent = DQNAgent(obs_space, action_space, config)
    action = agent.select_action(state, deterministic=False)
    loss = agent.update(batch)
"""

# Base interface
from .base_agent import BaseAgent

# Utilities (existing)
from .dqn_network import DQNNetwork, DuelingDQNNetwork
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

# Agents (refactored)
from .dqn import DQNAgent, DoubleDQNAgent

# Baseline agents
from .baseline import RSRPBaselineAgent

__all__ = [
    # Base
    'BaseAgent',
    # Agents
    'DQNAgent',
    'DoubleDQNAgent',
    # Baselines
    'RSRPBaselineAgent',
    # Utilities
    'DQNNetwork',
    'DuelingDQNNetwork',
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
]

__version__ = "2.0.0-refactor"
