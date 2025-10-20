#!/usr/bin/env python3
"""
Base Agent Interface

Unified interface for all RL agents in the satellite handover framework.

Design Principles:
- Standardized method names (select_action, update, save, load)
- Flexible parameter signatures (different algorithms have different update requirements)
- Algorithm-agnostic interface compatible with both off-policy and on-policy methods

Supported Algorithms:
- Off-policy: DQN, Double DQN, Dueling DQN, SAC
- On-policy: PPO, A2C (future support)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for all RL agents

    All agents must implement:
    - select_action(): Action selection with exploration/exploitation
    - update(): Agent learning update
    - save(): Model persistence
    - load(): Model loading

    Optional callbacks:
    - on_episode_start(): Called at episode start
    - on_episode_end(): Called at episode end
    - get_config(): Return agent configuration
    """

    @abstractmethod
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action given current state

        Args:
            state: Observation from environment (shape: [K, 12] for satellite handover)
            deterministic: If True, use greedy/deterministic policy (for evaluation)
                         If False, use exploration strategy (for training)

        Returns:
            action: Selected action index (int)

        Implementation Notes:
        - DQN: epsilon-greedy exploration when deterministic=False
        - PPO: sample from policy distribution when deterministic=False
        - A2C: sample from policy distribution when deterministic=False

        Example:
            >>> agent = DQNAgent(obs_space, action_space, config)
            >>> obs, info = env.reset()
            >>> action = agent.select_action(obs, deterministic=False)  # Training
            >>> action = agent.select_action(obs, deterministic=True)   # Evaluation
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Optional[float]:
        """
        Update agent based on experience

        NOTE: Different algorithms have different update signatures.
        This is intentional and allows each algorithm to use its natural interface.

        Common Signatures:
        - DQN (off-policy):
            update(batch: Dict) -> float
            OR
            update(state, action, reward, next_state, done) -> float

        - PPO (on-policy):
            update(trajectory: List[Dict]) -> float

        - A2C (on-policy):
            update(trajectory: List[Dict]) -> float

        Returns:
            loss: Training loss (float)
            OR None if not enough data to train yet

        Example (DQN):
            >>> loss = agent.update(batch)  # batch from replay buffer

        Example (PPO):
            >>> loss = agent.update(trajectory)  # collected episode trajectory
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save agent model to file

        Args:
            path: File path to save model (e.g., "checkpoints/model_ep100.pth")

        Implementation Requirements:
        - Save all necessary components (network weights, optimizer state, etc.)
        - Save agent-specific hyperparameters (epsilon, training_steps, etc.)
        - Use PyTorch's torch.save() for consistency

        Example:
            >>> agent.save("output/checkpoints/best_model.pth")
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load agent model from file

        Args:
            path: File path to load model from

        Implementation Requirements:
        - Restore all components saved by save()
        - Handle device mapping (CPU/GPU) correctly
        - Validate compatibility (architecture, dimensions)

        Example:
            >>> agent.load("output/checkpoints/best_model.pth")
            >>> # Continue training or evaluation
        """
        pass

    # ========== Optional Callback Methods ==========
    # Subclasses can override these for custom behavior

    def on_episode_start(self) -> None:
        """
        Called at the start of each training episode

        Use cases:
        - Reset episode-specific tracking variables
        - Initialize episode buffers (for on-policy methods)
        - Logging/debugging

        Default: No-op
        """
        pass

    def on_episode_end(self, episode_reward: float, episode_info: Dict[str, Any]) -> None:
        """
        Called at the end of each training episode

        Args:
            episode_reward: Total reward accumulated in the episode
            episode_info: Additional episode information (handovers, avg_rsrp, etc.)

        Use cases:
        - Update exploration parameters (e.g., epsilon decay)
        - Record episode statistics
        - Learning rate scheduling
        - Logging/debugging

        Default: No-op
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """
        Return agent configuration for logging/saving

        Returns:
            config: Dictionary of agent hyperparameters and settings

        Use cases:
        - Experiment tracking
        - Model reproducibility
        - Hyperparameter logging to TensorBoard

        Default: Empty dict

        Example:
            >>> config = agent.get_config()
            >>> # {'learning_rate': 1e-4, 'gamma': 0.99, 'epsilon': 0.1, ...}
        """
        return {}

    # ========== Utility Methods ==========

    def __repr__(self) -> str:
        """String representation of agent"""
        return f"{self.__class__.__name__}()"
