"""
Trainers Module - Training Logic for RL Algorithms

This module provides trainer classes that handle the training loop logic
for different types of RL algorithms.

Architecture:
    - OffPolicyTrainer: For algorithms with experience replay (DQN, SAC, etc.)
    - OnPolicyTrainer: For algorithms without replay (PPO, A2C, etc.) [Future]

Design Philosophy:
    - Trainers handle the training loop and environment interaction
    - Agents handle the algorithm-specific logic (network, updates, etc.)
    - This separation allows for modular, reusable components

Usage:
    from src.trainers import OffPolicyTrainer
    from src.agents import DQNAgent
    from src.environments import SatelliteHandoverEnv

    env = SatelliteHandoverEnv(...)
    agent = DQNAgent(...)
    trainer = OffPolicyTrainer(env, agent, config)

    for episode in range(num_episodes):
        metrics = trainer.train_episode(episode)
        print(f"Episode {episode}: reward={metrics['reward']}")
"""

from .off_policy_trainer import OffPolicyTrainer

# OnPolicyTrainer will be added in future phases if needed
# from .on_policy_trainer import OnPolicyTrainer

__all__ = [
    'OffPolicyTrainer',
]

__version__ = "2.0.0-refactor"
