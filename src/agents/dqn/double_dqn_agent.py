#!/usr/bin/env python3
"""
Double DQN Agent - Solves Q-value Overestimation

Key Improvement over Vanilla DQN:
- Decouples action selection from action evaluation
- Uses online network to SELECT best action
- Uses target network to EVALUATE that action
- Significantly reduces overestimation bias

Based on:
- "Deep Reinforcement Learning with Double Q-learning"
  van Hasselt, Guez, Silver (AAAI 2016)
- https://arxiv.org/abs/1509.06461

Inheritance:
- Extends DQNAgent (minimal changes needed)
- Only modifies the update() method for Double Q-learning
"""

import torch
import logging
from typing import Optional

from .dqn_agent import DQNAgent

logger = logging.getLogger(__name__)


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN Agent

    Inherits everything from DQNAgent except the Q-value update logic.

    Key Difference:

    Vanilla DQN:
        Q_target = r + γ * max_a' Q_target(s', a')
        Problem: max operation on same network causes overestimation

    Double DQN:
        a_max = argmax_a' Q_online(s', a')    ← Select action with online network
        Q_target = r + γ * Q_target(s', a_max) ← Evaluate with target network
        Solution: Decoupling reduces overestimation bias

    Usage:
        Same as DQNAgent - just replace algorithm name in config/train script
    """

    def __init__(self, observation_space, action_space, config):
        """
        Initialize Double DQN agent

        Args:
            observation_space: Gym observation space
            action_space: Gym action space
            config: Configuration dictionary (same as DQNAgent)
        """
        # Initialize parent DQNAgent (sets up networks, buffers, etc.)
        super().__init__(observation_space, action_space, config)

        logger.info("🎯 Using Double DQN (reduces Q-value overestimation)")

    def update(self, *args, **kwargs) -> Optional[float]:
        """
        Perform one Double DQN training step

        Returns:
            loss: Training loss (float)
            OR None if replay buffer doesn't have enough experiences

        Double DQN Update Formula:
            a_max = argmax_a' Q_online(s', a')
            Loss = MSE(Q(s,a), r + γ * Q_target(s', a_max))

        Key Difference from Vanilla DQN (line 274-277 in dqn_agent.py):
            Vanilla: max_next_q = Q_target(s').max()     ← max on target network
            Double:  a_max = Q_online(s').argmax()       ← argmax on online network
                     max_next_q = Q_target(s')[a_max]    ← evaluate on target network
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

        # ====== NUMERICAL STABILITY CHECK 1: Input Data ======
        if self.enable_nan_check:
            # Check for NaN/Inf in input data
            if torch.isnan(states).any() or torch.isinf(states).any():
                logger.error(f"[NaN/Inf Detection] NaN or Inf detected in states at step {self.training_steps}")
                logger.error(f"  States min: {states.min().item()}, max: {states.max().item()}")
                return None

            if torch.isnan(rewards).any() or torch.isinf(rewards).any():
                logger.error(f"[NaN/Inf Detection] NaN or Inf detected in rewards at step {self.training_steps}")
                logger.error(f"  Rewards min: {rewards.min().item()}, max: {rewards.max().item()}")
                return None

        # Current Q-values: Q(s, a)
        current_q_values = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # ====== NUMERICAL STABILITY CHECK 2: Q-values ======
        if self.enable_nan_check:
            if torch.isnan(current_q_values).any() or torch.isinf(current_q_values).any():
                logger.error(f"[NaN/Inf Detection] NaN or Inf detected in current Q-values at step {self.training_steps}")
                logger.error(f"  Q-values min: {current_q_values.min().item()}, max: {current_q_values.max().item()}")
                return None

        # Clip Q-values to prevent explosion
        current_q_values = torch.clamp(current_q_values, -self.q_value_clip, self.q_value_clip)

        # ==================== DOUBLE DQN LOGIC (KEY CHANGE) ====================
        # Step 1: Use ONLINE network to SELECT best action
        # Step 2: Use TARGET network to EVALUATE that action
        with torch.no_grad():
            # Online network selects best action: argmax_a' Q_online(s', a')
            next_q_values_online = self.q_network(next_states)
            next_actions = next_q_values_online.argmax(dim=1, keepdim=True)

            # Target network evaluates selected action: Q_target(s', a_max)
            next_q_values_target = self.target_network(next_states)
            max_next_q_values = next_q_values_target.gather(1, next_actions).squeeze(1)

            # ====== NUMERICAL STABILITY CHECK 3: Target Q-values ======
            if self.enable_nan_check:
                if torch.isnan(max_next_q_values).any() or torch.isinf(max_next_q_values).any():
                    logger.error(f"[NaN/Inf Detection] NaN or Inf detected in target Q-values at step {self.training_steps}")
                    logger.error(f"  Target Q min: {max_next_q_values.min().item()}, max: {max_next_q_values.max().item()}")
                    return None

            # Clip target Q-values
            max_next_q_values = torch.clamp(max_next_q_values, -self.q_value_clip, self.q_value_clip)

            # Target Q-values: r + γ * Q_target(s', a_max)
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

            # Clip final target to prevent explosion
            target_q_values = torch.clamp(target_q_values, -self.q_value_clip, self.q_value_clip)
        # ========================================================================

        # Compute loss (same as DQN)
        loss = self.criterion(current_q_values, target_q_values)

        # ====== NUMERICAL STABILITY CHECK 4: Loss ======
        if self.enable_nan_check:
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"[NaN/Inf Detection] NaN or Inf detected in loss at step {self.training_steps}")
                logger.error(f"  Loss value: {loss.item()}")
                return None

            # Warn if loss is abnormally large (but not infinite)
            if loss.item() > 1e6:
                logger.warning(f"[Large Loss Warning] Abnormally large loss detected: {loss.item():.2e} at step {self.training_steps}")
                logger.warning(f"  Current Q range: [{current_q_values.min().item():.2f}, {current_q_values.max().item():.2f}]")
                logger.warning(f"  Target Q range: [{target_q_values.min().item():.2f}, {target_q_values.max().item():.2f}]")
                logger.warning(f"  Rewards range: [{rewards.min().item():.2f}, {rewards.max().item():.2f}]")

        # Optimize (same as DQN)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.gradient_clip_norm)
        self.optimizer.step()

        # Update target network periodically (same as DQN)
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            logger.debug(f"Target network updated at step {self.training_steps}")

        # MEMORY FIX: Get loss value before tensors are released
        loss_value = loss.item()

        # MEMORY FIX: Explicitly delete tensors to prevent memory leak
        del states, actions, rewards, next_states, dones
        del current_q_values, target_q_values, loss
        del next_q_values_online, next_actions, next_q_values_target, max_next_q_values

        return loss_value
