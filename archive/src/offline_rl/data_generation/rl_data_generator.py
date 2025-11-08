#!/usr/bin/env python3
"""
RL Data Generator - Episode Generation for Training

Generates training episodes for satellite handover using orbit-engine adapter.

Features:
- 30-day dataset generation using TLE precision strategy
- Episode builder for orbital period windowing
- Integration with OrbitEngineAdapter for real physics
- Episode serialization and compression

Episode Structure:
    - States: 12-dimensional observations (RSRP, RSRQ, SINR, physical params, 3GPP offsets)
    - Actions: Ground truth optimal actions (computed from future states)
    - Timestamps: Episode timing information
    - Metadata: Satellite constellation, orbital parameters

Data Format:
    episodes/
    ├── episode_0001.npz
    ├── episode_0002.npz
    └── ...

    Each .npz file contains:
        - states: (T, 12) array
        - actions: (T,) array
        - rewards: (T,) array
        - next_states: (T, 12) array
        - dones: (T,) array
        - timestamps: (T,) array
        - metadata: dict

SOURCE:
- Episode generation strategy based on orbital mechanics
- Ground truth labeling using future state lookahead
- Data format compatible with DQN replay buffer
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import yaml

# Add parent directory for imports
CURRENT_DIR = Path(__file__).parent
SRC_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from adapters import OrbitEngineAdapter, TLELoader, TLE


class RLDataGenerator:
    """
    RL Data Generator for Satellite Handover.

    Generates training episodes using real satellite orbital data.

    Usage:
        generator = RLDataGenerator(config)
        generator.generate_dataset(start_date, end_date, output_dir)
    """

    def __init__(self, config: Dict):
        """
        Initialize RL Data Generator.

        Args:
            config: Configuration dictionary with:
                - data_generation.satellite_ids: List of satellite IDs
                - data_generation.time_step_seconds: Timestep between observations
                - data_generation.episode_duration_minutes: Episode duration
                - data_generation.ground_truth_lookahead_steps: Lookahead for labeling
                - data_generation.output_dir: Output directory for episodes
        """
        self.config = config
        self.data_gen_config = config.get('data_generation', {})

        # Satellite configuration
        self.satellite_ids = self.data_gen_config.get('satellite_ids', [])
        if not self.satellite_ids:
            raise ValueError("No satellite IDs specified in configuration")

        # Timing configuration
        self.time_step_seconds = self.data_gen_config.get('time_step_seconds', 5)
        self.episode_duration_minutes = self.data_gen_config.get('episode_duration_minutes', 95)
        self.episode_steps = int(self.episode_duration_minutes * 60 / self.time_step_seconds)

        # Ground truth labeling
        self.lookahead_steps = self.data_gen_config.get('ground_truth_lookahead_steps', 10)

        # Handover decision thresholds (SOURCE: 3GPP TS 38.133 v18.3.0)
        self.rsrp_threshold_dbm = self.data_gen_config.get('rsrp_threshold_dbm', -100.0)
        self.rsrp_hysteresis_db = self.data_gen_config.get('rsrp_hysteresis_db', 3.0)

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        # Initialize adapters
        print("🔧 Initializing OrbitEngineAdapter...")
        self.adapter = OrbitEngineAdapter(config)

        print(f"✅ RLDataGenerator initialized:")
        print(f"   Satellites: {len(self.satellite_ids)}")
        print(f"   Time step: {self.time_step_seconds}s")
        print(f"   Episode duration: {self.episode_duration_minutes}min ({self.episode_steps} steps)")
        print(f"   Lookahead: {self.lookahead_steps} steps")

    def generate_dataset(self,
                        start_date: datetime,
                        end_date: datetime,
                        output_dir: str,
                        max_episodes: Optional[int] = None) -> int:
        """
        Generate dataset for date range.

        Args:
            start_date: Start date for generation
            end_date: End date for generation
            output_dir: Output directory for episodes
            max_episodes: Maximum number of episodes (None = unlimited)

        Returns:
            num_episodes: Number of episodes generated
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("📊 Generating RL Training Dataset")
        print("=" * 70)
        print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Output directory: {output_dir}")
        print("=" * 70)

        # Generate episode start times
        episode_starts = self._generate_episode_windows(start_date, end_date, max_episodes)
        total_episodes = len(episode_starts)

        print(f"\n📦 Generating {total_episodes} episodes...")

        # Generate episodes
        generated_count = 0
        failed_count = 0

        for idx, episode_start in enumerate(tqdm(episode_starts, desc="Generating episodes")):
            try:
                episode_data = self._generate_episode(episode_start, idx)

                if episode_data is not None:
                    # Save episode
                    episode_path = output_path / f"episode_{idx:06d}.npz"
                    self._save_episode(episode_data, episode_path)
                    generated_count += 1
                else:
                    failed_count += 1

            except Exception as e:
                print(f"\n⚠️  Failed to generate episode {idx}: {e}")
                failed_count += 1
                continue

        print(f"\n✅ Dataset generation complete!")
        print(f"   Generated: {generated_count} episodes")
        print(f"   Failed: {failed_count} episodes")
        print(f"   Output: {output_dir}")

        return generated_count

    def _generate_episode_windows(self,
                                  start_date: datetime,
                                  end_date: datetime,
                                  max_episodes: Optional[int] = None) -> List[datetime]:
        """
        Generate episode start times within date range.

        Episodes are non-overlapping windows covering the entire period.

        Args:
            start_date: Start date
            end_date: End date
            max_episodes: Maximum number of episodes

        Returns:
            episode_starts: List of episode start times
        """
        episode_duration = timedelta(minutes=self.episode_duration_minutes)
        episode_starts = []

        current_time = start_date
        while current_time < end_date:
            episode_starts.append(current_time)
            current_time += episode_duration

            if max_episodes and len(episode_starts) >= max_episodes:
                break

        return episode_starts

    def _generate_episode(self, start_time: datetime, episode_id: int) -> Optional[Dict]:
        """
        Generate single episode.

        Args:
            start_time: Episode start time
            episode_id: Episode index

        Returns:
            episode_data: Dictionary with episode data, or None if failed
        """
        # Generate timestamps for episode
        timestamps = [
            start_time + timedelta(seconds=self.time_step_seconds * i)
            for i in range(self.episode_steps)
        ]

        # Select primary satellite (first in list for simplicity)
        # In production, this could be dynamic based on initial position
        primary_satellite = self.satellite_ids[0]

        # Calculate states for all timesteps
        # ✅ FIXED: Only keep valid connectable states, skip placeholders
        # SOURCE: fix.md P0 - Remove np.zeros() placeholder states
        states = []
        valid_timestamps = []
        valid_steps = 0

        for timestamp in timestamps:
            try:
                state = self.adapter.calculate_state(
                    satellite_id=primary_satellite,
                    timestamp=timestamp
                )

                # ✅ Only append valid connectable states
                if state and state.get('is_connectable', False):
                    # Convert state dict to array (12-dim)
                    state_array = self._state_dict_to_array(state)
                    states.append(state_array)
                    valid_timestamps.append(timestamp)
                    valid_steps += 1
                # ✅ REMOVED: No placeholder - skip non-connectable periods entirely
                # ❌ OLD: states.append(np.zeros(12, dtype=np.float32))
                # Rationale: Zero vectors are not real physical states and violate
                # "NO MOCK/SIMULATION DATA" principle (CLAUDE.md)

            except Exception as e:
                # ✅ FIXED: Skip erroneous states instead of using placeholder
                # Log error for debugging
                self.logger.warning(
                    f"Failed to calculate state for {primary_satellite} at {timestamp}: {e}"
                )
                # ❌ OLD: states.append(np.zeros(12, dtype=np.float32))
                # ✅ NEW: Skip and continue
                continue

        # ✅ FIXED: Require at least 5% valid steps (realistic for LEO satellites)
        # SOURCE: LEO satellite visibility is ~10% per orbital period (10min pass / 95min orbit)
        # Rationale: 50% threshold was unrealistic - satellites spend most time below horizon
        if valid_steps < self.episode_steps * 0.05:
            return None

        # Update timestamps to match valid states only
        timestamps = valid_timestamps

        states = np.array(states, dtype=np.float32)

        # ✅ FIXED: Episode length is now dynamic (valid_steps), not fixed (episode_steps)
        actual_episode_length = len(states)

        # Generate ground truth actions using lookahead
        actions = self._generate_ground_truth_actions(states, timestamps, primary_satellite)

        # Calculate rewards (for offline RL, can compute from transitions)
        rewards = self._calculate_episode_rewards(states, actions)

        # Generate next_states and dones
        next_states = np.roll(states, -1, axis=0)
        next_states[-1] = states[-1]  # Last state

        # ✅ FIXED: Use actual_episode_length instead of episode_steps
        dones = np.zeros(actual_episode_length, dtype=np.float32)
        dones[-1] = 1.0  # Episode ends at last step

        # Package episode data
        episode_data = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'timestamps': np.array([t.timestamp() for t in timestamps], dtype=np.float64),
            'metadata': {
                'episode_id': episode_id,
                'start_time': start_time.isoformat(),
                'primary_satellite': primary_satellite,
                'valid_steps': valid_steps,  # Number of valid connectable states
                'actual_steps': actual_episode_length,  # ✅ NEW: Actual episode length (after removing placeholders)
                'requested_steps': self.episode_steps,  # ✅ RENAMED: Originally requested episode length
                'time_step_seconds': self.time_step_seconds,
                'coverage_rate': valid_steps / self.episode_steps  # ✅ NEW: Percentage of valid coverage
            }
        }

        return episode_data

    def _state_dict_to_array(self, state: Dict) -> np.ndarray:
        """
        Convert state dictionary to 12-dimensional array.

        Args:
            state: State dictionary from adapter

        Returns:
            state_array: 12-dimensional numpy array
        """
        return np.array([
            state.get('rsrp_dbm', -120.0),
            state.get('rsrq_db', -20.0),
            state.get('rs_sinr_db', -5.0),
            state.get('distance_km', 2000.0),
            state.get('elevation_deg', 0.0),
            state.get('doppler_shift_hz', 0.0),
            state.get('radial_velocity_ms', 0.0),
            state.get('atmospheric_loss_db', 0.0),
            state.get('path_loss_db', 160.0),
            state.get('propagation_delay_ms', 20.0),
            state.get('offset_mo_db', 0.0),
            state.get('cell_offset_db', 0.0)
        ], dtype=np.float32)

    def _generate_ground_truth_actions(self,
                                       states: np.ndarray,
                                       timestamps: List[datetime],
                                       primary_satellite: str) -> np.ndarray:
        """
        Generate ground truth actions using future state lookahead.

        Strategy:
            - Look ahead N steps
            - If RSRP will degrade significantly, label as handover (1)
            - Otherwise, label as maintain (0)

        Args:
            states: State array (T, 12)
            timestamps: Timestamp list
            primary_satellite: Current satellite ID

        Returns:
            actions: Action array (T,) with 0=maintain, 1=handover
        """
        T = len(states)
        actions = np.zeros(T, dtype=np.int64)

        for t in range(T):
            # Current RSRP (first dimension)
            current_rsrp = states[t, 0]

            # Look ahead
            if t + self.lookahead_steps < T:
                future_rsrp = states[t + self.lookahead_steps, 0]

                # Decision: Handover if RSRP will degrade significantly
                # SOURCE: 3GPP TS 38.133 v18.3.0 (handover decision thresholds)
                rsrp_degradation = current_rsrp - future_rsrp

                if rsrp_degradation > self.rsrp_hysteresis_db:
                    # RSRP degrading, should handover
                    actions[t] = 1
                elif current_rsrp < self.rsrp_threshold_dbm:
                    # Below threshold, should handover
                    actions[t] = 1
                else:
                    # Maintain current satellite
                    actions[t] = 0
            else:
                # Near end of episode, maintain
                actions[t] = 0

        return actions

    def _calculate_episode_rewards(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Calculate rewards for episode transitions.

        Uses same reward function as BaseHandoverEnvironment.

        Args:
            states: State array (T, 12)
            actions: Action array (T,)

        Returns:
            rewards: Reward array (T,)
        """
        T = len(states)
        rewards = np.zeros(T, dtype=np.float32)

        # Reward weights (from config or defaults)
        w_qos = 1.0
        w_handover = 0.5
        w_signal = 0.3
        w_ping_pong = 1.0

        last_handover_step = -1

        for t in range(T):
            action = actions[t]
            current_rsrp = states[t, 0]

            # QoS improvement (if handover)
            qos_improvement = 0.0
            if action == 1 and t + 1 < T:
                next_rsrp = states[t + 1, 0]
                rsrp_diff = next_rsrp - current_rsrp
                qos_improvement = np.clip(rsrp_diff / 60.0, -1.0, 1.0)

            # Handover penalty
            handover_penalty = 1.0 if action == 1 else 0.0

            # Signal quality (normalized reward based on RSRP)
            # SOURCE: 3GPP TS 38.133 - RSRP range typically -44 to -140 dBm
            # Good signal: -90 dBm, Poor: -110 dBm, Critical: -120 dBm
            if current_rsrp > -85:
                signal_quality = 1.0  # Excellent
            elif current_rsrp > -95:
                signal_quality = 0.5  # Good
            elif current_rsrp > -105:
                signal_quality = 0.0  # Fair
            elif current_rsrp > -115:
                signal_quality = -0.5  # Poor
            else:
                signal_quality = -1.0  # Critical

            # Ping-pong penalty
            ping_pong_penalty = 0.0
            if action == 1 and last_handover_step >= 0:
                if t - last_handover_step < 10:
                    ping_pong_penalty = 1.0

            # Total reward
            rewards[t] = (
                w_qos * qos_improvement
                - w_handover * handover_penalty
                + w_signal * signal_quality
                - w_ping_pong * ping_pong_penalty
            )

            # Update last handover step
            if action == 1:
                last_handover_step = t

        return rewards

    def _save_episode(self, episode_data: Dict, filepath: Path):
        """
        Save episode to compressed numpy file.

        Args:
            episode_data: Episode data dictionary
            filepath: Output file path
        """
        np.savez_compressed(filepath, **episode_data)

    @staticmethod
    def load_episode(filepath: str) -> Dict:
        """
        Load episode from file.

        Args:
            filepath: Episode file path

        Returns:
            episode_data: Episode data dictionary
        """
        data = np.load(filepath, allow_pickle=True)

        return {
            'states': data['states'],
            'actions': data['actions'],
            'rewards': data['rewards'],
            'next_states': data['next_states'],
            'dones': data['dones'],
            'timestamps': data['timestamps'],
            'metadata': data['metadata'].item()
        }


# Example usage
if __name__ == "__main__":
    print("RL Data Generator for Satellite Handover")
    print("=" * 60)

    # Load configuration
    config_path = Path(__file__).parent.parent.parent / "config" / "data_gen_config.yaml"

    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print(f"✅ Configuration loaded: {config_path}")

        # Create generator
        generator = RLDataGenerator(config)

        # Generate 1-day validation dataset
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 2, 0, 0, 0)
        output_dir = "data/episodes/validation_1day"

        print(f"\n📊 Generating validation dataset...")
        num_episodes = generator.generate_dataset(
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            max_episodes=10  # Limit for testing
        )

        print(f"\n✅ Generated {num_episodes} episodes")

    else:
        print(f"⚠️  Configuration file not found: {config_path}")
        print("Using dummy configuration for testing...")

        # Dummy config
        config = {
            'data_generation': {
                'satellite_ids': ['STARLINK-1007'],
                'time_step_seconds': 5,
                'episode_duration_minutes': 95,
                'ground_truth_lookahead_steps': 10,
                'rsrp_threshold_dbm': -100.0,
                'rsrp_hysteresis_db': 3.0
            }
        }

        print("⚠️  Cannot test without orbit-engine adapter")
