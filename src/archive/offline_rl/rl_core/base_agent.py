#!/usr/bin/env python3
"""
Base RL Agent - Algorithm-Agnostic Interface

Defines the standard interface that all RL agents must implement.

Supported Algorithms:
- DQN (Deep Q-Network)
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- CQL (Conservative Q-Learning) - Offline RL
- IQL (Implicit Q-Learning) - Offline RL

Design Pattern: Abstract Base Class
- Ensures all agents have consistent interface
- Enables UniversalRLTrainer to work with any algorithm
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np


class BaseRLAgent(ABC):
    """
    Base class for all RL agents.

    All concrete agent implementations (DQN, PPO, SAC, etc.) must inherit
    from this class and implement the required methods.

    Attributes:
        state_dim: Dimension of state space (12 for handover problem)
        action_dim: Dimension of action space (2: maintain/handover)
        config: Algorithm-specific configuration dictionary
    """

    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        """
        Initialize base agent.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            config: Algorithm-specific configuration
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # Training state
        self.total_steps = 0
        self.total_episodes = 0

    @abstractmethod
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Select an action given a state.

        Args:
            state: Current state observation (numpy array of shape [state_dim])
            eval_mode: If True, use greedy/deterministic policy (no exploration)

        Returns:
            action: Selected action (integer 0 or 1)
                0 = maintain (keep current satellite)
                1 = handover (switch to best neighbor)

        Note:
            - In training mode (eval_mode=False): May use exploration (ε-greedy, entropy, etc.)
            - In evaluation mode (eval_mode=True): Use greedy/deterministic policy
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Union[float, Dict[str, float]]:
        """
        Update agent parameters (learning step).

        Args:
            *args, **kwargs: Algorithm-specific arguments

        Returns:
            loss: Training loss(es) for logging
                - Single float: Total loss
                - Dict: Multiple loss components (e.g., {'policy_loss': 0.5, 'value_loss': 0.3})

        Algorithm-specific signatures:

        DQN:
            update() -> float
            Uses internal replay buffer, no arguments needed

        PPO:
            update(trajectories: List[Dict]) -> Dict[str, float]
            Needs full trajectories with states, actions, rewards, advantages

        SAC:
            update() -> Dict[str, float]
            Uses internal replay buffer, returns multiple loss components

        CQL/IQL (Offline):
            update(batch: Dict) -> Dict[str, float]
            Samples from fixed offline dataset
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save agent checkpoint to disk.

        Args:
            path: Checkpoint file path (e.g., "models/dqn_best.pth")

        Should save:
            - Network parameters (actor, critic, Q-networks, etc.)
            - Optimizer state
            - Training state (epsilon, step count, etc.)
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Load agent checkpoint from disk.

        Args:
            path: Checkpoint file path

        Should restore:
            - Network parameters
            - Optimizer state
            - Training state
        """
        pass

    # Optional methods (can be overridden by subclasses)

    def on_episode_start(self):
        """
        Called at the start of each episode.

        Use for:
            - Resetting episode-specific state
            - Logging
            - Curriculum learning adjustments
        """
        self.total_episodes += 1

    def on_episode_end(self, episode_reward: float, episode_length: int):
        """
        Called at the end of each episode.

        Args:
            episode_reward: Total reward for the episode
            episode_length: Number of steps in the episode

        Use for:
            - Logging episode statistics
            - Updating learning rate schedules
            - Updating exploration schedules
        """
        pass

    def on_step(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        """
        Called after each environment step.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended

        Use for:
            - Storing transition in replay buffer
            - Online learning updates
            - Logging
        """
        self.total_steps += 1

    def get_training_state(self) -> Dict:
        """
        Get current training state for logging/checkpointing.

        Returns:
            Dictionary with training state:
                - total_steps: Total training steps
                - total_episodes: Total training episodes
                - algorithm-specific state (epsilon, learning rate, etc.)
        """
        return {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes
        }

    def set_training_mode(self, mode: bool):
        """
        Set training mode.

        Args:
            mode: True for training, False for evaluation

        Use for:
            - Enabling/disabling dropout
            - Enabling/disabling batch normalization updates
            - Switching between exploration and exploitation
        """
        pass

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"state_dim={self.state_dim}, "
                f"action_dim={self.action_dim}, "
                f"steps={self.total_steps})")


class DummyAgent(BaseRLAgent):
    """
    Dummy agent for testing (random policy).

    This is a minimal implementation for testing the framework.
    Real agents (DQN, PPO, etc.) should be implemented in src/agents/.
    """

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Select random action"""
        return np.random.randint(0, self.action_dim)

    def update(self, *args, **kwargs) -> float:
        """No-op update (dummy)"""
        return 0.0

    def save(self, path: str):
        """No-op save (dummy)"""
        pass

    def load(self, path: str):
        """No-op load (dummy)"""
        pass


# Example usage
if __name__ == "__main__":
    print("BaseRLAgent - Algorithm-Agnostic Interface")
    print("=" * 60)

    # Create dummy agent for testing
    agent = DummyAgent(state_dim=12, action_dim=2, config={})

    # Test action selection
    state = np.random.randn(12)
    action = agent.select_action(state)
    print(f"✅ Action selection: {action}")

    # Test update
    loss = agent.update()
    print(f"✅ Update: loss={loss}")

    # Test training state
    training_state = agent.get_training_state()
    print(f"✅ Training state: {training_state}")

    print("\n" + "=" * 60)
    print("BaseRLAgent interface verified!")
    print("\nSupported algorithms:")
    print("  - DQN (Deep Q-Network)")
    print("  - PPO (Proximal Policy Optimization)")
    print("  - SAC (Soft Actor-Critic)")
    print("  - CQL (Conservative Q-Learning)")
    print("  - IQL (Implicit Q-Learning)")
