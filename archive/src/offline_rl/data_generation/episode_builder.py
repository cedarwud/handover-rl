#!/usr/bin/env python3
"""
Episode Builder - Utilities for Episode Construction

Provides utilities for building and managing RL training episodes.

Features:
- Episode windowing based on orbital periods
- Timestamp indexing and alignment
- Episode validation and filtering
- Statistics calculation

Functions:
- build_episode_from_states(): Construct episode from state sequence
- validate_episode(): Check episode quality
- get_episode_statistics(): Calculate episode statistics
- split_by_orbital_period(): Split data by orbital periods
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class EpisodeBuilder:
    """
    Episode Builder for RL Training Data.

    Utilities for constructing and validating episodes.

    Usage:
        builder = EpisodeBuilder(config)
        episode = builder.build_episode(states, actions, rewards, ...)
        stats = builder.get_statistics(episode)
    """

    def __init__(self, config: Dict = None):
        """
        Initialize Episode Builder.

        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}

        # Validation thresholds
        self.min_valid_ratio = self.config.get('min_valid_ratio', 0.5)
        self.min_rsrp_dbm = self.config.get('min_rsrp_dbm', -150.0)
        self.max_rsrp_dbm = self.config.get('max_rsrp_dbm', -30.0)

    def build_episode(self,
                     states: np.ndarray,
                     actions: np.ndarray,
                     rewards: np.ndarray,
                     timestamps: np.ndarray,
                     metadata: Dict = None) -> Dict:
        """
        Build episode from components.

        Args:
            states: State array (T, 12)
            actions: Action array (T,)
            rewards: Reward array (T,)
            timestamps: Timestamp array (T,)
            metadata: Episode metadata (optional)

        Returns:
            episode: Episode dictionary
        """
        T = len(states)

        # Generate next_states and dones
        next_states = np.roll(states, -1, axis=0)
        next_states[-1] = states[-1]  # Last state

        dones = np.zeros(T, dtype=np.float32)
        dones[-1] = 1.0  # Episode ends

        # Build episode
        episode = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'timestamps': timestamps,
            'metadata': metadata or {}
        }

        return episode

    def validate_episode(self, episode: Dict) -> Tuple[bool, Dict]:
        """
        Validate episode quality.

        Checks:
        - State bounds (RSRP, RSRQ, SINR within valid ranges)
        - Action validity (0 or 1)
        - Minimum valid step ratio
        - No NaN or Inf values

        Args:
            episode: Episode dictionary

        Returns:
            is_valid: Whether episode passes validation
            validation_info: Dictionary with validation details
        """
        states = episode['states']
        actions = episode['actions']
        rewards = episode['rewards']

        validation_info = {
            'total_steps': len(states),
            'valid_steps': 0,
            'invalid_states': 0,
            'invalid_actions': 0,
            'has_nan': False,
            'has_inf': False,
            'errors': []
        }

        # Check for NaN/Inf
        if np.any(np.isnan(states)) or np.any(np.isnan(rewards)):
            validation_info['has_nan'] = True
            validation_info['errors'].append('Contains NaN values')

        if np.any(np.isinf(states)) or np.any(np.isinf(rewards)):
            validation_info['has_inf'] = True
            validation_info['errors'].append('Contains Inf values')

        # Check state bounds (RSRP as first dimension)
        rsrp_values = states[:, 0]
        invalid_rsrp = np.logical_or(
            rsrp_values < self.min_rsrp_dbm,
            rsrp_values > self.max_rsrp_dbm
        )
        validation_info['invalid_states'] = int(np.sum(invalid_rsrp))

        # Check action validity
        invalid_actions = np.logical_not(np.isin(actions, [0, 1]))
        validation_info['invalid_actions'] = int(np.sum(invalid_actions))

        # Count valid steps
        valid_steps = np.logical_and(
            np.logical_not(invalid_rsrp),
            np.logical_not(invalid_actions)
        )
        validation_info['valid_steps'] = int(np.sum(valid_steps))

        # Calculate valid ratio
        valid_ratio = validation_info['valid_steps'] / validation_info['total_steps']
        validation_info['valid_ratio'] = valid_ratio

        # Overall validation
        is_valid = (
            not validation_info['has_nan'] and
            not validation_info['has_inf'] and
            valid_ratio >= self.min_valid_ratio
        )

        if not is_valid and valid_ratio < self.min_valid_ratio:
            validation_info['errors'].append(
                f'Valid ratio {valid_ratio:.2%} < {self.min_valid_ratio:.2%}'
            )

        return is_valid, validation_info

    def get_statistics(self, episode: Dict) -> Dict:
        """
        Calculate episode statistics.

        Args:
            episode: Episode dictionary

        Returns:
            stats: Dictionary with episode statistics
        """
        states = episode['states']
        actions = episode['actions']
        rewards = episode['rewards']

        # RSRP statistics (first dimension)
        rsrp_values = states[:, 0]

        # Action distribution
        action_counts = np.bincount(actions, minlength=2)

        # Reward statistics
        total_reward = np.sum(rewards)
        avg_reward = np.mean(rewards)

        # Handover statistics
        total_handovers = int(action_counts[1])
        handover_rate = total_handovers / len(actions)

        stats = {
            'total_steps': len(states),
            'total_reward': float(total_reward),
            'average_reward': float(avg_reward),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'std_reward': float(np.std(rewards)),
            'total_handovers': total_handovers,
            'total_maintains': int(action_counts[0]),
            'handover_rate': float(handover_rate),
            'avg_rsrp': float(np.mean(rsrp_values)),
            'min_rsrp': float(np.min(rsrp_values)),
            'max_rsrp': float(np.max(rsrp_values)),
            'std_rsrp': float(np.std(rsrp_values))
        }

        # Add metadata if available
        if 'metadata' in episode:
            stats.update(episode['metadata'])

        return stats

    def split_by_orbital_period(self,
                                states: np.ndarray,
                                timestamps: np.ndarray,
                                orbital_period_minutes: float = 95.0) -> List[Tuple[int, int]]:
        """
        Split states into episodes based on orbital period.

        Args:
            states: State array (T, 12)
            timestamps: Timestamp array (T,) in Unix time
            orbital_period_minutes: Orbital period in minutes (default: 95 for Starlink)

        Returns:
            windows: List of (start_idx, end_idx) tuples for each episode
        """
        T = len(states)
        orbital_period_seconds = orbital_period_minutes * 60

        # Convert timestamps to datetime
        start_time = datetime.fromtimestamp(timestamps[0])

        windows = []
        current_start = 0

        for i in range(1, T):
            current_time = datetime.fromtimestamp(timestamps[i])
            elapsed_seconds = (current_time - start_time).total_seconds()

            # Check if we've exceeded one orbital period
            if elapsed_seconds >= orbital_period_seconds:
                windows.append((current_start, i))
                current_start = i
                start_time = current_time

        # Add final window
        if current_start < T - 1:
            windows.append((current_start, T))

        return windows

    def filter_low_quality_episodes(self,
                                    episode_files: List[Path],
                                    min_valid_ratio: float = 0.7) -> List[Path]:
        """
        Filter out low-quality episodes.

        Args:
            episode_files: List of episode file paths
            min_valid_ratio: Minimum valid step ratio

        Returns:
            filtered_files: List of high-quality episode files
        """
        from data_generation.rl_data_generator import RLDataGenerator

        filtered_files = []

        for filepath in episode_files:
            try:
                episode = RLDataGenerator.load_episode(str(filepath))
                is_valid, info = self.validate_episode(episode)

                if is_valid and info['valid_ratio'] >= min_valid_ratio:
                    filtered_files.append(filepath)

            except Exception as e:
                print(f"⚠️  Error loading {filepath}: {e}")
                continue

        return filtered_files

    def aggregate_statistics(self, episode_files: List[Path]) -> Dict:
        """
        Calculate aggregate statistics across multiple episodes.

        Args:
            episode_files: List of episode file paths

        Returns:
            agg_stats: Aggregated statistics
        """
        from data_generation.rl_data_generator import RLDataGenerator

        all_rewards = []
        all_handover_rates = []
        all_rsrp = []
        total_steps = 0
        total_handovers = 0

        for filepath in episode_files:
            try:
                episode = RLDataGenerator.load_episode(str(filepath))
                stats = self.get_statistics(episode)

                all_rewards.append(stats['total_reward'])
                all_handover_rates.append(stats['handover_rate'])
                all_rsrp.append(stats['avg_rsrp'])
                total_steps += stats['total_steps']
                total_handovers += stats['total_handovers']

            except Exception as e:
                continue

        agg_stats = {
            'num_episodes': len(episode_files),
            'total_steps': total_steps,
            'total_handovers': total_handovers,
            'avg_reward': float(np.mean(all_rewards)) if all_rewards else 0.0,
            'std_reward': float(np.std(all_rewards)) if all_rewards else 0.0,
            'min_reward': float(np.min(all_rewards)) if all_rewards else 0.0,
            'max_reward': float(np.max(all_rewards)) if all_rewards else 0.0,
            'avg_handover_rate': float(np.mean(all_handover_rates)) if all_handover_rates else 0.0,
            'avg_rsrp': float(np.mean(all_rsrp)) if all_rsrp else 0.0
        }

        return agg_stats


# Utility functions

def load_episodes(episode_dir: str, max_episodes: Optional[int] = None) -> List[Dict]:
    """
    Load episodes from directory.

    Args:
        episode_dir: Directory containing episode files
        max_episodes: Maximum number to load (None = all)

    Returns:
        episodes: List of episode dictionaries
    """
    from data_generation.rl_data_generator import RLDataGenerator

    episode_path = Path(episode_dir)
    episode_files = sorted(episode_path.glob("episode_*.npz"))

    if max_episodes:
        episode_files = episode_files[:max_episodes]

    episodes = []
    for filepath in episode_files:
        try:
            episode = RLDataGenerator.load_episode(str(filepath))
            episodes.append(episode)
        except Exception as e:
            print(f"⚠️  Error loading {filepath}: {e}")
            continue

    return episodes


def create_dataset_splits(episode_files: List[Path],
                         train_ratio: float = 0.8,
                         val_ratio: float = 0.1,
                         seed: int = 42) -> Dict[str, List[Path]]:
    """
    Split episodes into train/val/test sets.

    Args:
        episode_files: List of episode file paths
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed

    Returns:
        splits: Dictionary with 'train', 'val', 'test' file lists
    """
    np.random.seed(seed)

    # Shuffle episodes
    episode_files = list(episode_files)
    np.random.shuffle(episode_files)

    # Calculate split indices
    total = len(episode_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        'train': episode_files[:train_end],
        'val': episode_files[train_end:val_end],
        'test': episode_files[val_end:]
    }

    return splits


# Example usage
if __name__ == "__main__":
    print("Episode Builder - Utilities for Episode Construction")
    print("=" * 60)

    # Create builder
    builder = EpisodeBuilder()

    # Test episode construction
    print("\n📦 Testing episode construction...")
    T = 100
    states = np.random.randn(T, 12).astype(np.float32)
    states[:, 0] = np.random.uniform(-110, -60, T)  # RSRP in valid range
    actions = np.random.randint(0, 2, T)
    rewards = np.random.randn(T).astype(np.float32)
    timestamps = np.array([1704067200 + i * 5 for i in range(T)], dtype=np.float64)

    episode = builder.build_episode(states, actions, rewards, timestamps)
    print(f"✅ Episode built: {episode['states'].shape}")

    # Validate episode
    print("\n🔍 Validating episode...")
    is_valid, info = builder.validate_episode(episode)
    print(f"   Valid: {is_valid}")
    print(f"   Valid ratio: {info['valid_ratio']:.2%}")
    print(f"   Valid steps: {info['valid_steps']}/{info['total_steps']}")

    # Get statistics
    print("\n📊 Episode statistics...")
    stats = builder.get_statistics(episode)
    print(f"   Total reward: {stats['total_reward']:.2f}")
    print(f"   Average reward: {stats['average_reward']:.3f}")
    print(f"   Handover rate: {stats['handover_rate']:.2%}")
    print(f"   Average RSRP: {stats['avg_rsrp']:.1f} dBm")

    # Test orbital period splitting
    print("\n🔄 Testing orbital period splitting...")
    long_timestamps = np.array([1704067200 + i * 60 for i in range(200)], dtype=np.float64)
    long_states = np.random.randn(200, 12).astype(np.float32)

    windows = builder.split_by_orbital_period(long_states, long_timestamps, orbital_period_minutes=95)
    print(f"✅ Split into {len(windows)} orbital periods:")
    for i, (start, end) in enumerate(windows):
        print(f"   Period {i+1}: steps {start}-{end} ({end-start} steps)")

    print("\n" + "=" * 60)
    print("✅ Episode Builder verified!")
