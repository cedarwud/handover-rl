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
from gymnasium.vector import VectorEnv

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
            # Use action mask from info if available
            action_mask = info.get('action_mask', None)
            action = self.agent.select_action(obs, deterministic=False, action_mask=action_mask)

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

    def train_episode_vectorized(
        self,
        episode_idx: int,
        episode_start_time: Optional[Any] = None,
        seed: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Train for one episode using vectorized environments

        This method handles multiple environments running in parallel.
        Episodes are collected from all environments simultaneously.

        Args:
            episode_idx: Episode index (for logging and seed)
            episode_start_time: Optional start time for environment reset
            seed: Random seed for this episode

        Returns:
            metrics: Aggregated metrics from all parallel environments
        """
        num_envs = self.env.num_envs

        # Callback: Episode start
        self.agent.on_episode_start()

        # Reset all environments
        reset_options = {}
        if episode_start_time is not None:
            reset_options['start_time'] = episode_start_time

        if seed is not None:
            # Each environment gets a different seed
            seeds = [seed + i for i in range(num_envs)]
            obs, infos = self.env.reset(seed=seeds, options=reset_options)
        else:
            obs, infos = self.env.reset(options=reset_options)

        # Episode state (track for each environment)
        episode_rewards = np.zeros(num_envs)
        episode_steps = np.zeros(num_envs, dtype=int)
        episode_losses = []
        dones = np.zeros(num_envs, dtype=bool)

        # Keep track of which environments are still running
        active_envs = np.ones(num_envs, dtype=bool)

        # Episode loop (continues until all environments are done)
        while active_envs.any():
            # 1. Agent selects actions for all active environments
            # Note: obs is shape (num_envs, ...)
            # Extract action masks from infos (if available)
            if isinstance(infos, dict) and 'action_mask' in infos:
                # Dict-style infos (Gymnasium vectorized envs)
                action_masks = infos['action_mask']  # Shape: (num_envs, action_space.n)
            else:
                # List-style infos or no action_mask
                action_masks = [None] * num_envs

            actions = np.array([
                self.agent.select_action(obs[i], deterministic=False, action_mask=action_masks[i])
                if active_envs[i] else 0  # Dummy action for done envs
                for i in range(num_envs)
            ])

            # 2. All environments execute actions
            next_obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
            dones = np.logical_or(terminateds, truncateds)

            # 3. Store experiences from all environments
            for i in range(num_envs):
                if active_envs[i]:  # Only store from active environments
                    if hasattr(self.agent, 'store_experience'):
                        self.agent.store_experience(
                            obs[i], actions[i], rewards[i], next_obs[i], dones[i]
                        )

                    # Update metrics for this environment
                    episode_rewards[i] += rewards[i]
                    episode_steps[i] += 1
                    self.total_steps += 1

                    # Mark environment as done if finished
                    if dones[i]:
                        active_envs[i] = False

            # 4. Per-step update (can update after any environment step)
            if self.total_steps % self.update_frequency == 0:
                loss = self.agent.update()
                if loss is not None:
                    episode_losses.append(loss)
                    self.total_updates += 1

            obs = next_obs

        # All environments finished, aggregate metrics
        # infos is a dict with keys that map to arrays (one value per env)
        # We need to aggregate these metrics across all environments

        # Extract metrics from all environments and compute means
        if isinstance(infos, dict):
            # Dict-style infos (Gymnasium vectorized envs)
            num_handovers_array = infos.get('num_handovers', np.zeros(num_envs))
            avg_rsrp_array = infos.get('avg_rsrp', np.zeros(num_envs))
            num_ping_pongs_array = infos.get('num_ping_pongs', np.zeros(num_envs))

            # Aggregate across environments
            avg_handovers = float(np.mean(num_handovers_array)) if isinstance(num_handovers_array, np.ndarray) else float(num_handovers_array)
            avg_rsrp = float(np.mean(avg_rsrp_array)) if isinstance(avg_rsrp_array, np.ndarray) else float(avg_rsrp_array)
            avg_ping_pongs = float(np.mean(num_ping_pongs_array)) if isinstance(num_ping_pongs_array, np.ndarray) else float(num_ping_pongs_array)
        else:
            # List-style infos (one dict per environment)
            avg_handovers = float(np.mean([info.get('num_handovers', 0) for info in infos]))
            avg_rsrp = float(np.mean([info.get('avg_rsrp', 0) for info in infos]))
            avg_ping_pongs = float(np.mean([info.get('num_ping_pongs', 0) for info in infos]))

        episode_info = {
            'num_handovers': avg_handovers,
            'avg_rsrp': avg_rsrp,
            'num_ping_pongs': avg_ping_pongs,
        }
        self.agent.on_episode_end(float(np.mean(episode_rewards)), episode_info)

        # Compile aggregated metrics
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0

        metrics = {
            'reward': float(np.mean(episode_rewards)),  # Average across environments
            'length': int(np.mean(episode_steps)),
            'loss': float(avg_loss),
            'handovers': avg_handovers,
            'avg_rsrp': avg_rsrp,
            'ping_pongs': avg_ping_pongs,
            'num_updates': len(episode_losses),
        }

        return metrics

    def __repr__(self) -> str:
        """String representation"""
        env_name = 'Vectorized' if isinstance(self.env, VectorEnv) else (self.env.spec.id if self.env.spec else 'Unknown')
        return (f"OffPolicyTrainer(env={env_name}, "
                f"agent={self.agent.__class__.__name__}, "
                f"steps={self.total_steps}, updates={self.total_updates})")
