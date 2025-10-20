#!/usr/bin/env python3
"""
Multi-Satellite Handover Environment

Academic Standard: Real TLE data, Complete physics, No hardcoding
Literature: Graph RL (Aerospace 2024), DHO Protocol (IEEE TWC 2023)

Based on Graph RL paper multi-satellite architecture:
- Query ALL satellites at each timestep
- Select top-K by RSRP
- Dynamic action space (select which satellite)
- Multi-objective reward
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SatelliteHandoverEnv(gym.Env):
    """
    LEO Satellite Handover Environment

    Implements multi-satellite state representation and dynamic action space
    based on Graph RL paper (Aerospace 2024) methodology.

    State Space:
        Box(shape=(K, 12), dtype=float32)
        K = max_visible_satellites (e.g., 10)
        12 = state dimensions per satellite

    Action Space:
        Discrete(K+1)
        0 = stay with current satellite
        1 to K = switch to candidate[i-1]

    Reward:
        Multi-objective: QoS + handover_penalty + ping_pong_penalty
    """

    metadata = {'render_modes': []}

    def __init__(self, adapter, satellite_ids: List[str], config: Dict):
        """
        Initialize environment

        Args:
            adapter: OrbitEngineAdapter instance (real physics)
            satellite_ids: List of satellite IDs to query (e.g., 125 satellites)
            config: Configuration dictionary

        Academic Compliance:
            - adapter must use real TLE data (Space-Track.org)
            - satellite_ids from real constellation
            - config contains no hardcoded physics values
        """
        super().__init__()

        # Store adapter (OrbitEngineAdapter with complete ITU-R/3GPP physics)
        self.adapter = adapter
        self.satellite_ids = satellite_ids
        self.config = config

        # Parameters from config (no hardcoding)
        gs_config = config.get('ground_station', {})
        self.min_elevation_deg = gs_config.get('min_elevation_deg', 10.0)

        env_config = config.get('environment', config.get('data_generation', {}))
        self.time_step_seconds = env_config.get('time_step_seconds', 5)
        self.episode_duration_minutes = env_config.get('episode_duration_minutes', 95)
        self.max_visible_satellites = env_config.get('max_visible_satellites', 10)

        # Reward function weights (from config or defaults based on Graph RL paper)
        reward_config = env_config.get('reward', {})
        self.reward_weights = {
            'qos': reward_config.get('qos_weight', 1.0),
            'handover_penalty': reward_config.get('handover_penalty', -0.1),
            'ping_pong_penalty': reward_config.get('ping_pong_penalty', -0.2),
        }

        # Observation space: (K, 12) matrix
        # K satellites × 12 features per satellite
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_visible_satellites, 12),
            dtype=np.float32
        )

        # Action space: Discrete(K+1)
        # 0 = stay, 1-K = switch to candidate[i-1]
        self.action_space = spaces.Discrete(self.max_visible_satellites + 1)

        # Environment state
        self.current_time = None
        self.current_satellite = None
        self.episode_start = None
        self.previous_satellite = None
        self.current_visible_satellites = []  # List of satellite IDs in current observation
        self.current_visible_states = []  # List of state dicts for current observation
        self.handover_history = []  # Track handover sequence for ping-pong detection

        # Statistics
        self.episode_stats = {
            'num_handovers': 0,
            'num_ping_pongs': 0,
            'avg_rsrp': 0.0,
            'timesteps': 0,
        }

        logger.info(f"SatelliteHandoverEnv initialized")
        logger.info(f"  Satellite pool: {len(self.satellite_ids)} satellites")
        logger.info(f"  Max visible: {self.max_visible_satellites}")
        logger.info(f"  Observation space: {self.observation_space}")
        logger.info(f"  Action space: {self.action_space}")

    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state

        Args:
            seed: Random seed for reproducibility
            options: Optional dict with 'start_time' key

        Returns:
            observation: (K, 12) state matrix
            info: Dict with episode information
        """
        super().reset(seed=seed)

        # Set episode start time
        if options and 'start_time' in options:
            self.episode_start = options['start_time']
        else:
            # Default to TLE data range (2025-10-07)
            self.episode_start = datetime(2025, 10, 7, 0, 0, 0)

        self.current_time = self.episode_start
        self.previous_satellite = None
        self.handover_history = []

        # Reset statistics
        self.episode_stats = {
            'num_handovers': 0,
            'num_ping_pongs': 0,
            'avg_rsrp': 0.0,
            'timesteps': 0,
        }

        # Get initial observation
        observation = self._get_observation()

        # Select initial satellite (highest RSRP)
        if len(self.current_visible_satellites) > 0:
            self.current_satellite = self.current_visible_satellites[0]
            self.handover_history.append(self.current_satellite)
        else:
            # No satellites visible - episode will end immediately
            self.current_satellite = None

        info = {
            'current_satellite': self.current_satellite,
            'num_visible': len(self.current_visible_satellites),
            'episode_start': self.episode_start.isoformat(),
            'current_time': self.current_time.isoformat(),
        }

        logger.debug(f"Environment reset - {info['num_visible']} visible satellites")

        return observation, info

    def step(self, action: int):
        """
        Execute one timestep

        Args:
            action: Integer from action space
                    0 = stay with current satellite
                    1 to K = switch to candidate[i-1]

        Returns:
            observation: Next state
            reward: Reward for this transition
            terminated: Episode ended
            truncated: Episode truncated (time limit)
            info: Additional information
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}, must be in {self.action_space}")

        # Record previous satellite for reward calculation
        prev_satellite = self.current_satellite
        handover_occurred = False

        # Execute action
        if action == 0:
            # Action 0: Stay with current satellite
            # Check if current satellite still available
            if self.current_satellite not in self.current_visible_satellites:
                # Current satellite lost - forced handover to best available
                if len(self.current_visible_satellites) > 0:
                    self.current_satellite = self.current_visible_satellites[0]
                    handover_occurred = True
                    logger.debug(f"Forced handover: current satellite lost")
                else:
                    # No satellites available - episode will terminate
                    self.current_satellite = None
        else:
            # Action 1-K: Switch to candidate satellite
            candidate_idx = action - 1

            if candidate_idx < len(self.current_visible_satellites):
                new_satellite = self.current_visible_satellites[candidate_idx]

                if new_satellite != self.current_satellite:
                    # Handover to new satellite
                    self.previous_satellite = self.current_satellite
                    self.current_satellite = new_satellite
                    handover_occurred = True

                    # Track handover history for ping-pong detection
                    self.handover_history.append(new_satellite)
                    if len(self.handover_history) > 10:
                        self.handover_history = self.handover_history[-10:]

                    logger.debug(f"Handover: {prev_satellite} -> {new_satellite}")
            else:
                # Invalid action index (out of range) - treat as stay
                logger.warning(f"Action {action} out of range (only {len(self.current_visible_satellites)} visible)")

        # Advance time
        self.current_time += timedelta(seconds=self.time_step_seconds)

        # Get next observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward(
            observation=observation,
            handover_occurred=handover_occurred,
            prev_sat=prev_satellite,
            curr_sat=self.current_satellite
        )

        # Check if episode is done
        terminated, truncated = self._check_done()

        # Update statistics
        self.episode_stats['timesteps'] += 1
        if handover_occurred:
            self.episode_stats['num_handovers'] += 1

        # Calculate average RSRP (for stats)
        if self.current_satellite and len(self.current_visible_satellites) > 0:
            curr_idx = self.current_visible_satellites.index(self.current_satellite) \
                if self.current_satellite in self.current_visible_satellites else 0
            if curr_idx < len(self.current_visible_states):
                curr_rsrp = self.current_visible_states[curr_idx].get('rsrp_dbm', 0)
                # Running average
                self.episode_stats['avg_rsrp'] = (
                    (self.episode_stats['avg_rsrp'] * (self.episode_stats['timesteps'] - 1) + curr_rsrp) /
                    self.episode_stats['timesteps']
                )

        # Build info dict
        info = {
            'current_satellite': self.current_satellite,
            'num_visible': len(self.current_visible_satellites),
            'handover_occurred': handover_occurred,
            'current_time': self.current_time.isoformat(),
            **self.episode_stats,
        }

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Generate multi-satellite observation

        Based on Graph RL paper:
        - Query ALL satellites at each timestep
        - Select top-K by RSRP
        - Return (K, 12) matrix

        Returns:
            observation: (max_visible_satellites, 12) array

        Academic Compliance:
            - Uses OrbitEngineAdapter (real TLE + complete physics)
            - No hardcoded values
            - No mock data
        """
        visible_satellites = []

        # CRITICAL: Query ALL satellites (Graph RL paper methodology)
        # This is the key difference from old single-satellite approach
        for sat_id in self.satellite_ids:
            try:
                # Real physics calculation via OrbitEngineAdapter
                # Uses complete ITU-R P.676-13 + 3GPP TS 38.214/215
                state_dict = self.adapter.calculate_state(
                    satellite_id=sat_id,
                    timestamp=self.current_time
                )

                if not state_dict:
                    continue

                # Check visibility (elevation >= threshold)
                elevation = state_dict.get('elevation_deg', 0)
                if elevation < self.min_elevation_deg:
                    continue

                # Check connectivity (3GPP standards)
                if not state_dict.get('is_connectable', False):
                    continue

                # Convert to 12-dimensional vector
                state_vector = self._state_dict_to_vector(state_dict)

                visible_satellites.append({
                    'id': sat_id,
                    'state': state_vector,
                    'rsrp': state_dict['rsrp_dbm'],
                    'elevation': elevation,
                    'distance': state_dict.get('distance_km', 0),
                    'state_dict': state_dict,  # Keep for reward calculation
                })

            except Exception as e:
                # Log error but continue querying other satellites
                logger.debug(f"Error querying {sat_id}: {e}")
                continue

        # Sort by RSRP (best signal first) - Graph RL paper methodology
        visible_satellites.sort(key=lambda x: x['rsrp'], reverse=True)

        # Take top-K candidates
        top_satellites = visible_satellites[:self.max_visible_satellites]

        # Build observation matrix: (K, 12)
        observation = np.zeros(
            (self.max_visible_satellites, 12),
            dtype=np.float32
        )

        for i, sat_info in enumerate(top_satellites):
            observation[i] = sat_info['state']

        # Store current candidates (for action mapping)
        # Action i maps to satellite i in this list
        self.current_visible_satellites = [s['id'] for s in top_satellites]
        self.current_visible_states = [s['state_dict'] for s in top_satellites]

        logger.debug(f"Observation generated: {len(top_satellites)}/{len(visible_satellites)} "
                     f"top satellites from {len(self.satellite_ids)} total")

        return observation

    def _state_dict_to_vector(self, state_dict: Dict) -> np.ndarray:
        """
        Convert state dict to 12-dimensional vector

        No hardcoding - use actual values from OrbitEngineAdapter

        State dimensions (verified from OrbitEngineAdapter):
        0: RSRP (dBm) - rsrp_dbm
        1: RSRQ (dB) - rsrq_db
        2: SINR (dB) - rs_sinr_db
        3: Distance (km) - distance_km
        4: Elevation (deg) - elevation_deg
        5: Doppler shift (Hz) - doppler_shift_hz
        6: Path loss (dB) - path_loss_db
        7: Atmospheric loss (dB) - atmospheric_loss_db
        8: Radial velocity (m/s) - radial_velocity_ms
        9: Offset MO (dB) - offset_mo_db
        10: Cell offset (dB) - cell_offset_db
        11: Propagation delay (ms) - propagation_delay_ms

        Args:
            state_dict: State from OrbitEngineAdapter

        Returns:
            state_vector: 12-dimensional numpy array
        """
        return np.array([
            state_dict.get('rsrp_dbm', 0),
            state_dict.get('rsrq_db', 0),
            state_dict.get('rs_sinr_db', 0),
            state_dict.get('distance_km', 0),
            state_dict.get('elevation_deg', 0),
            state_dict.get('doppler_shift_hz', 0),
            state_dict.get('path_loss_db', 0),
            state_dict.get('atmospheric_loss_db', 0),
            state_dict.get('radial_velocity_ms', 0),
            state_dict.get('offset_mo_db', 0),
            state_dict.get('cell_offset_db', 0),
            state_dict.get('propagation_delay_ms', 0)
        ], dtype=np.float32)

    def _calculate_reward(self, observation, handover_occurred, prev_sat, curr_sat):
        """
        Calculate multi-objective reward

        Based on Graph RL paper reward structure:
        - QoS component (RSRP-based)
        - Handover penalty
        - Ping-pong penalty

        Args:
            observation: Current observation matrix
            handover_occurred: Whether handover happened this step
            prev_sat: Previous satellite ID
            curr_sat: Current satellite ID

        Returns:
            reward: Scalar reward value
        """
        reward = 0.0

        # Component 1: QoS reward (based on current RSRP)
        # Higher RSRP = better signal quality = higher reward
        if curr_sat and len(self.current_visible_satellites) > 0:
            try:
                curr_idx = self.current_visible_satellites.index(curr_sat)
                if curr_idx < len(self.current_visible_states):
                    curr_rsrp = self.current_visible_states[curr_idx].get('rsrp_dbm', -140)

                    # Normalize RSRP to [0, 1] range
                    # 3GPP range: -140 to -44 dBm
                    # Map to: 0 (worst) to 1 (best)
                    rsrp_normalized = (curr_rsrp + 140) / ((-44) - (-140))
                    rsrp_normalized = np.clip(rsrp_normalized, 0.0, 1.0)

                    # QoS reward weighted
                    qos_reward = rsrp_normalized * self.reward_weights['qos']
                    reward += qos_reward
                else:
                    # No valid state - small penalty
                    reward -= 0.1
            except (ValueError, IndexError):
                # Satellite not in visible list - penalty
                reward -= 0.1
        else:
            # No current satellite - large penalty
            reward -= 1.0

        # Component 2: Handover penalty
        # Penalize handovers to encourage stability
        if handover_occurred:
            reward += self.reward_weights['handover_penalty']

        # Component 3: Ping-pong penalty
        # Penalize switching back to recent satellite
        if handover_occurred and len(self.handover_history) >= 3:
            # Check if we're ping-ponging (returning to a recent satellite)
            recent_sats = self.handover_history[-3:]
            if len(set(recent_sats)) < len(recent_sats):
                # Repeated satellite in recent history = ping-pong
                reward += self.reward_weights['ping_pong_penalty']
                self.episode_stats['num_ping_pongs'] += 1

        return float(reward)

    def _check_done(self) -> Tuple[bool, bool]:
        """
        Check if episode is done

        Returns:
            terminated: True if episode naturally ended
            truncated: True if episode hit time limit
        """
        # Time limit reached
        episode_duration = timedelta(minutes=self.episode_duration_minutes)
        if self.current_time >= self.episode_start + episode_duration:
            return False, True  # Not terminated, but truncated

        # No satellites visible
        if len(self.current_visible_satellites) == 0:
            return True, False  # Terminated (no connectivity)

        # Current satellite not in visible list
        if self.current_satellite not in self.current_visible_satellites:
            return True, False  # Terminated (lost current satellite)

        return False, False  # Continue

    def render(self):
        """Render environment state (optional)"""
        pass  # Not implemented - terminal logging sufficient

    def close(self):
        """Clean up resources"""
        pass  # No resources to clean up
