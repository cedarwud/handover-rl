#!/usr/bin/env python3
"""
Off-Policy Trainer

Trainer for off-policy RL algorithms (DQN, Double DQN, SAC, etc.)

Key Features:
- Experience replay buffer
- Per-step updates (can update after every environment step)
- Can learn from old experiences (off-policy property)
- Supports any agent implementing BaseAgent interface

Supported Algorithms:
- DQN (Deep Q-Network)
- Double DQN
- Dueling DQN
- SAC (Soft Actor-Critic)

Based on:
- Standard DQN training loop (Mnih et al., Nature 2015)
- Multi-satellite handover application (Graph RL, Aerospace 2024)
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
import gymnasium as gym

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class OffPolicyTrainer:
    """
    Trainer for off-policy RL algorithms

    Implements the standard off-policy training loop:
    1. Agent selects action (with exploration)
    2. Environment executes action
    3. Store experience in replay buffer
    4. Per-step update from replay buffer
    5. Repeat until episode terminates

    This trainer is algorithm-agnostic and works with any BaseAgent
    implementation that supports off-policy learning.
    """

    def __init__(
        self,
        env: gym.Env,
        agent: BaseAgent,
        config: Dict[str, Any]
    ):
        """
        Initialize off-policy trainer

        Args:
            env: Gymnasium environment (SatelliteHandoverEnv)
            agent: Agent instance implementing BaseAgent interface
            config: Training configuration dictionary

        Config Parameters:
            - min_buffer_size: Minimum replay buffer size before training starts
            - batch_size: Batch size for training updates
            - update_frequency: How often to update (1 = every step)
        """
        self.env = env
        self.agent = agent
        self.config = config

        # Extract training parameters
        agent_config = config.get('agent', {})
        self.min_buffer_size = agent_config.get('min_buffer_size', 64)
        self.batch_size = agent_config.get('batch_size', 64)
        self.update_frequency = agent_config.get('update_frequency', 1)

        # Training statistics
        self.total_steps = 0
        self.total_updates = 0

        logger.info("OffPolicyTrainer initialized")
        logger.info(f"  Min buffer size: {self.min_buffer_size}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Update frequency: {self.update_frequency}")

    def train_episode(
        self,
        episode_idx: int,
        episode_start_time: Optional[Any] = None,
        seed: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Train for one episode

        Args:
            episode_idx: Episode index (for logging and seed)
            episode_start_time: Optional start time for environment reset
            seed: Random seed for this episode

        Returns:
            metrics: Dictionary containing:
                - reward: Total episode reward
                - length: Episode length (number of steps)
                - loss: Average training loss
                - handovers: Number of handovers
                - avg_rsrp: Average RSRP (dBm)
                - ping_pongs: Number of ping-pong handovers
                - num_updates: Number of training updates performed

        Training Flow:
            1. Reset environment
            2. Agent callback: on_episode_start()
            3. Episode loop:
               - Agent selects action (with exploration)
               - Environment step
               - Store experience in replay buffer
               - Per-step update (if buffer has enough data)
            4. Agent callback: on_episode_end()
            5. Return episode metrics
        """
        # Callback: Episode start
        self.agent.on_episode_start()

        # Reset environment
        reset_options = {}
        if episode_start_time is not None:
            reset_options['start_time'] = episode_start_time

        if seed is not None:
            obs, info = self.env.reset(seed=seed, options=reset_options)
        else:
            obs, info = self.env.reset(options=reset_options)

        # Episode state
        episode_reward = 0.0
        episode_steps = 0
        episode_losses = []
        done = False

        # Episode loop
        while not done:
            # 1. Agent selects action (training mode = exploration)
            action = self.agent.select_action(obs, deterministic=False)

            # 2. Environment executes action
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # 3. Store experience in agent's replay buffer
            # Note: Agent is responsible for managing its own replay buffer
            # This supports different buffer implementations (standard, prioritized, etc.)
            if hasattr(self.agent, 'store_experience'):
                self.agent.store_experience(obs, action, reward, next_obs, done)

            # 4. Per-step update (off-policy characteristic)
            # Check if we have enough experiences and it's time to update
            if self.total_steps % self.update_frequency == 0:
                # Agent decides if it's ready to train (e.g., buffer size check)
                loss = self.agent.update()

                if loss is not None:
                    episode_losses.append(loss)
                    self.total_updates += 1

            # Update metrics
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            obs = next_obs

        # Callback: Episode end
        episode_info = {
            'num_handovers': info.get('num_handovers', 0),
            'avg_rsrp': info.get('avg_rsrp', 0),
            'num_ping_pongs': info.get('num_ping_pongs', 0),
        }
        self.agent.on_episode_end(episode_reward, episode_info)

        # Compile episode metrics
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0

        metrics = {
            'reward': float(episode_reward),
            'length': episode_steps,
            'loss': float(avg_loss),
            'handovers': info.get('num_handovers', 0),
            'avg_rsrp': info.get('avg_rsrp', 0),
            'ping_pongs': info.get('num_ping_pongs', 0),
            'num_updates': len(episode_losses),
        }

        return metrics

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get trainer statistics

        Returns:
            stats: Dictionary with training statistics
                - total_steps: Total environment steps
                - total_updates: Total training updates
                - update_ratio: Updates per step ratio
        """
        update_ratio = self.total_updates / self.total_steps if self.total_steps > 0 else 0.0

        return {
            'total_steps': self.total_steps,
            'total_updates': self.total_updates,
            'update_ratio': update_ratio,
        }

    def __repr__(self) -> str:
        """String representation"""
        return (f"OffPolicyTrainer(env={self.env.spec.id if self.env.spec else 'Unknown'}, "
                f"agent={self.agent.__class__.__name__}, "
                f"steps={self.total_steps}, updates={self.total_updates})")
