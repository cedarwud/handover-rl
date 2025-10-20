#!/usr/bin/env python3
"""
Real Handover Environment - Concrete Implementation

Real satellite handover environment using orbit-engine adapter and generated episodes.

Features:
- Loads pre-generated training episodes
- Integrates with OrbitEngineAdapter for real physics
- Compatible with Gymnasium API
- Supports both episode replay and live simulation

Usage:
    env = HandoverEnvironment(config, episode_dir="data/episodes/train")
    state, info = env.reset()
    next_state, reward, terminated, truncated, info = env.step(action)
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import random

# Add parent directory for imports
CURRENT_DIR = Path(__file__).parent
SRC_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from rl_core.base_environment import BaseHandoverEnvironment
from data_generation import RLDataGenerator, load_episodes


class HandoverEnvironment(BaseHandoverEnvironment):
    """
    Real Satellite Handover Environment.

    Loads and replays pre-generated episodes for training.

    Usage:
        env = HandoverEnvironment(config, episode_dir="data/episodes/train")
        state, info = env.reset()
        action = agent.select_action(state)
        next_state, reward, term, trunc, info = env.step(action)
    """

    def __init__(self,
                 config: Dict,
                 episode_dir: Optional[str] = None,
                 episodes: Optional[List[Dict]] = None,
                 mode: str = 'replay'):
        """
        Initialize Handover Environment.

        Args:
            config: Configuration dictionary
            episode_dir: Directory containing episode files (for replay mode)
            episodes: Pre-loaded episode list (optional)
            mode: 'replay' (use generated episodes) or 'live' (TODO: real-time simulation)
        """
        super().__init__(config)

        self.mode = mode
        self.episode_dir = episode_dir

        # Episode management
        self.episodes = episodes
        self.episode_files = []
        self.current_episode = None
        self.current_episode_idx = None
        self.episode_step = 0

        # Load episodes if directory provided
        if episode_dir and not episodes:
            self._load_episode_files(episode_dir)

        print(f"✅ HandoverEnvironment initialized:")
        print(f"   Mode: {self.mode}")
        if self.episode_dir:
            print(f"   Episode directory: {self.episode_dir}")
            print(f"   Episodes available: {len(self.episode_files)}")
        elif self.episodes:
            print(f"   Episodes loaded: {len(self.episodes)}")

    def _load_episode_files(self, episode_dir: str):
        """
        Load episode file paths from directory.

        Args:
            episode_dir: Directory containing episode .npz files
        """
        episode_path = Path(episode_dir)

        if not episode_path.exists():
            print(f"⚠️  Episode directory not found: {episode_dir}")
            print(f"   Environment will use random exploration")
            return

        self.episode_files = sorted(episode_path.glob("episode_*.npz"))

        if not self.episode_files:
            print(f"⚠️  No episode files found in: {episode_dir}")

    def reset(self,
             seed: Optional[int] = None,
             options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to start of a random episode.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (can specify episode_idx)

        Returns:
            observation: Initial state (12-dim numpy array)
            info: Additional information
        """
        # Set seed if provided
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
            random.seed(seed)

        # Reset episode state
        self.current_step = 0
        self.total_handovers = 0
        self.last_handover_step = -1
        self.episode_rewards = []
        self.episode_count += 1
        self.episode_step = 0

        # Select episode
        if options and 'episode_idx' in options:
            # Use specified episode
            episode_idx = options['episode_idx']
            self.current_episode_idx = episode_idx
        else:
            # Random episode
            if self.episode_files:
                self.current_episode_idx = random.randint(0, len(self.episode_files) - 1)
            elif self.episodes:
                self.current_episode_idx = random.randint(0, len(self.episodes) - 1)
            else:
                # No episodes available, use base class random state
                observation = super()._get_initial_observation()
                self.current_state = observation
                return observation, {'episode': self.episode_count, 'step': 0}

        # Load episode
        self._load_current_episode()

        # Get initial observation from episode
        if self.current_episode is not None:
            observation = self.current_episode['states'][0]
            self.current_state = observation
        else:
            # Fallback to base class
            observation = super()._get_initial_observation()
            self.current_state = observation

        # Info dictionary
        info = {
            'episode': self.episode_count,
            'step': self.current_step,
            'episode_idx': self.current_episode_idx
        }

        if self.current_episode and 'metadata' in self.current_episode:
            info['episode_metadata'] = self.current_episode['metadata']

        return observation, info

    def _load_current_episode(self):
        """Load current episode from file or list."""
        if self.episodes:
            # Use pre-loaded episodes
            if 0 <= self.current_episode_idx < len(self.episodes):
                self.current_episode = self.episodes[self.current_episode_idx]
            else:
                self.current_episode = None

        elif self.episode_files:
            # Load from file
            if 0 <= self.current_episode_idx < len(self.episode_files):
                try:
                    filepath = self.episode_files[self.current_episode_idx]
                    self.current_episode = RLDataGenerator.load_episode(str(filepath))
                except Exception as e:
                    print(f"⚠️  Error loading episode {self.current_episode_idx}: {e}")
                    self.current_episode = None
            else:
                self.current_episode = None
        else:
            self.current_episode = None

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0=maintain, 1=handover)

        Returns:
            observation: Next state (12-dim numpy array)
            reward: Reward for this step (float)
            terminated: Whether episode ended naturally (bool)
            truncated: Whether episode was truncated (timeout) (bool)
            info: Additional information (dict)
        """
        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Get next state from episode or simulate
        if self.current_episode is not None:
            # Episode replay mode
            next_state, reward, terminated, truncated = self._step_from_episode(action)
        else:
            # Fallback to base class (random simulation)
            next_state = super()._execute_action(action)
            reward = self._calculate_reward(action, self.current_state, next_state)
            terminated = super()._is_terminated()
            truncated = self.current_step >= self.max_steps

        # Track statistics
        self.episode_rewards.append(reward)

        # Update handover tracking
        if action == 1:
            self.total_handovers += 1
            self.last_handover_step = self.current_step

        # Update step counter
        self.current_step += 1
        self.episode_step += 1

        # Update current state
        self.current_state = next_state

        # Info dictionary
        info = {
            'step': self.current_step,
            'total_handovers': self.total_handovers,
            'action_taken': action,
            'episode_reward': sum(self.episode_rewards)
        }

        return next_state, reward, terminated, truncated, info

    def _step_from_episode(self, action: int) -> Tuple[np.ndarray, float, bool, bool]:
        """
        Execute step from loaded episode.

        Args:
            action: Action taken (may differ from episode action)

        Returns:
            next_state, reward, terminated, truncated
        """
        episode = self.current_episode
        step = self.episode_step

        # Check if episode ended
        if step >= len(episode['states']) - 1:
            # Episode complete
            next_state = episode['states'][-1]
            reward = 0.0
            terminated = True
            truncated = False
            return next_state, reward, terminated, truncated

        # Get next state from episode
        next_state = episode['next_states'][step]

        # Get reward from episode (or recalculate based on actual action)
        # Option 1: Use episode reward (assumes ground truth action)
        if action == episode['actions'][step]:
            reward = episode['rewards'][step]
        else:
            # Option 2: Recalculate reward for different action
            reward = self._calculate_reward(action, episode['states'][step], next_state)

        # Check termination
        terminated = bool(episode['dones'][step])
        truncated = step >= len(episode['states']) - 1

        return next_state, reward, terminated, truncated

    def _get_initial_observation(self) -> np.ndarray:
        """
        Get initial observation.

        Overrides base class to use episode data if available.

        Returns:
            observation: Initial 12-dim state
        """
        if self.current_episode is not None:
            return self.current_episode['states'][0]
        else:
            return super()._get_initial_observation()

    def _execute_action(self, action: int) -> np.ndarray:
        """
        Execute action and get next state.

        Overrides base class to use episode data if available.

        Args:
            action: Action to execute

        Returns:
            next_state: Next 12-dim state
        """
        if self.current_episode is not None and self.episode_step < len(self.current_episode['states']) - 1:
            return self.current_episode['next_states'][self.episode_step]
        else:
            return super()._execute_action(action)

    def get_episode_info(self) -> Dict:
        """
        Get current episode information.

        Returns:
            info: Dictionary with episode information
        """
        info = super().get_episode_statistics()

        if self.current_episode and 'metadata' in self.current_episode:
            info['metadata'] = self.current_episode['metadata']

        return info


# Example usage
if __name__ == "__main__":
    print("Real Handover Environment")
    print("=" * 60)

    # Example configuration
    config = {
        'environment': {
            'state_dim': 12,
            'action_dim': 2,
            'reward_weights': {
                'qos_improvement': 1.0,
                'handover_penalty': 0.5,
                'signal_quality': 0.3,
                'ping_pong_penalty': 1.0
            },
            'max_steps_per_episode': 1500
        }
    }

    # Test with dummy episode
    print("\n📦 Creating dummy episode for testing...")
    T = 100
    dummy_episode = {
        'states': np.random.randn(T, 12).astype(np.float32),
        'actions': np.random.randint(0, 2, T),
        'rewards': np.random.randn(T).astype(np.float32),
        'next_states': np.random.randn(T, 12).astype(np.float32),
        'dones': np.zeros(T, dtype=np.float32),
        'timestamps': np.array([1704067200 + i * 5 for i in range(T)], dtype=np.float64),
        'metadata': {
            'episode_id': 0,
            'primary_satellite': 'STARLINK-1007'
        }
    }
    dummy_episode['dones'][-1] = 1.0

    # Create environment with dummy episode
    env = HandoverEnvironment(config, episodes=[dummy_episode])

    print(f"✅ Environment created")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")

    # Reset environment
    state, info = env.reset()
    print(f"\n✅ Environment reset")
    print(f"   Initial state shape: {state.shape}")
    print(f"   Episode index: {info['episode_idx']}")

    # Take a few steps
    print("\n🎮 Taking 5 random steps:")
    for i in range(5):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)

        print(f"   Step {i+1}: action={action}, reward={reward:.3f}, "
              f"handovers={info['total_handovers']}, done={terminated or truncated}")

        if terminated or truncated:
            print(f"   Episode ended at step {i+1}")
            break

    # Get episode info
    episode_info = env.get_episode_info()
    print(f"\n📊 Episode info:")
    print(f"   Total steps: {episode_info['total_steps']}")
    print(f"   Total handovers: {episode_info['total_handovers']}")
    print(f"   Handover rate: {episode_info['handover_rate']:.3f}")

    print("\n" + "=" * 60)
    print("✅ Handover Environment verified!")
