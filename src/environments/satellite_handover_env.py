#!/usr/bin/env python3
"""
Multi-Satellite Handover Environment

RVT-Based Reward Function (Academic Standard)
=================================================
Reference: "User-Centric Satellite Handover for Multiple Traffic Profiles
Using Deep Q-Learning" (IEEE TAES 2024, Equation 14)

Previous Issue (V2-V8): QoS-based rewards led to extreme stay behavior (99.4% stay).
With 10-minute episodes, RSRP doesn't change much, so agent learns "never switch".

Solution: RVT-Based Reward (Paper Equation 14)
- Handover to loaded satellite: -500 (z1)
- Handover to free satellite: -300 (z2)
- Stay on loaded satellite: -100 * load_factor (f1)
- Stay on free satellite: +RVT (f2, Remaining Visibility Time in seconds)

Key Innovation: RVT (Remaining Visibility Time)
- Predicts when satellite will drop below min_elevation
- Encourages switching before satellite loss
- Natural incentive for proactive handovers
- Paper result: 4.2 handovers/episode, 0.033% blocking

Expected Results (from paper):
- Handovers: ~4.2 per episode (600s, 120 steps)
- Blocking rate: ~0.033%
- Balanced stay/switch behavior
- Training: 2500 episodes

Academic Standard: Real TLE data, Complete physics, RVT-based rewards
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

    Key Improvement: RVT-based reward function following IEEE TAES 2024 paper.

    State Space:
        Box(shape=(K, 14), dtype=float32)
        K = max_visible_satellites (e.g., 15)
        14 = state dimensions per satellite (added RVT dimension)

    Action Space:
        Discrete(K+1)
        0 = stay with current satellite
        1 to K = switch to candidate[i-1] (FIXED for entire episode)

    Reward (IEEE TAES 2024 Equation 14):
        If handover:
            - To loaded satellite: -500 (z1)
            - To free satellite: -300 (z2)
        If stay:
            - On loaded satellite: -100 * load_factor (f1)
            - On free satellite: +RVT (f2, Remaining Visibility Time)
    """

    metadata = {'render_modes': []}

    def __init__(self, adapter, satellite_ids: List[str], config: Dict):
        """
        Initialize environment

        Args:
            adapter: OrbitEngineAdapter instance (real physics)
            satellite_ids: List of satellite IDs to query
            config: Configuration dictionary
        """
        super().__init__()

        self.adapter = adapter
        self.satellite_ids = satellite_ids
        self.config = config

        # Parameters from config
        gs_config = config.get('ground_station', {})
        self.min_elevation_deg = gs_config.get('min_elevation_deg', 20.0)

        env_config = config.get('environment', config.get('data_generation', {}))
        self.time_step_seconds = env_config.get('time_step_seconds', 5)
        self.episode_duration_minutes = env_config.get('episode_duration_minutes', 10)
        self.max_visible_satellites = env_config.get('max_visible_satellites', 15)

        # V9: RVT-based reward weights (from IEEE TAES 2024 paper)
        reward_config = env_config.get('reward', {})
        self.reward_weights = {
            # QoS component (minimal, not primary reward)
            'qos': reward_config.get('qos_weight', 0.1),
            'sinr_weight': reward_config.get('sinr_weight', 0.0),
            'latency_weight': reward_config.get('latency_weight', 0.0),

            # Handover penalties (from paper)
            'handover_to_loaded': reward_config.get('handover_to_loaded_penalty', -500.0),  # z1
            'handover_to_free': reward_config.get('handover_to_free_penalty', -300.0),      # z2

            # Stay rewards (from paper)
            'stay_loaded_factor': reward_config.get('stay_loaded_penalty_factor', 100.0),  # f1
            'rvt_reward_weight': reward_config.get('rvt_reward_weight', 1.0),              # f2

            # Legacy (kept for compatibility)
            'ping_pong_penalty': reward_config.get('ping_pong_penalty', -50.0),
            'connectivity_bonus': reward_config.get('connectivity_bonus', 0.0),
            'stability_bonus': reward_config.get('stability_bonus', 0.0),  # Not used in V9
        }

        # V9: RVT calculation parameters
        self.rvt_lookahead_minutes = 60  # Look ahead 60 minutes to find horizon crossing
        self.rvt_check_interval_seconds = 30  # Check every 30 seconds

        # Satellite load tracking (simplified: based on recent handover rate)
        self.satellite_load = {}  # sat_id -> load_factor (0.0 to 1.0)
        self.load_threshold = 0.7  # Above this is considered "loaded"

        # Observation space: (K, 14) matrix (added RVT dimension)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_visible_satellites, 14),  # 13 + 1 RVT dimension
            dtype=np.float32
        )

        # Action space: Discrete(K+1)
        self.action_space = spaces.Discrete(self.max_visible_satellites + 1)

        # Environment state
        self.current_time = None
        self.current_satellite = None
        self.episode_start = None
        self.previous_satellite = None
        self.handover_history = []

        # V9: RVT cache
        self.rvt_cache = {}  # sat_id -> (timestamp, rvt_seconds)
        self.rvt_cache_duration = 60  # Cache valid for 60 seconds

        # V9+: Minimum Dwell Time Constraint (Physical/Protocol constraint)
        # Based on 3GPP NTN standards: Handover preparation ~15-20s minimum
        # Academic research typically uses 30-60s
        # This prevents unrealistic frequent switching
        self.min_dwell_time_seconds = reward_config.get('min_dwell_time_seconds', 60)
        self.min_dwell_time_steps = max(1, int(self.min_dwell_time_seconds / self.time_step_seconds))
        self.steps_since_last_handover = 0  # Counter for dwell time enforcement

        # Episode-fixed candidate set
        self.episode_candidate_ids = []
        self.episode_candidate_states = []
        self.episode_candidate_available = []

        # Statistics
        self.episode_stats = {
            'num_handovers': 0,
            'num_ping_pongs': 0,
            'avg_rsrp': 0.0,
            'timesteps': 0,
            'connectivity_ratio': 0.0,
            'connected_steps': 0,
            'avg_rvt': 0.0,  # V9: Track average RVT
            'total_reward': 0.0,  # V9: Track cumulative reward
        }

        logger.info(f"SatelliteHandoverEnv initialized (RVT-based reward)")
        logger.info(f"  Satellite pool: {len(self.satellite_ids)} satellites")
        logger.info(f"  Max candidates per episode: {self.max_visible_satellites}")
        logger.info(f"  Min elevation: {self.min_elevation_deg}°")
        logger.info(f"  V9 Reward: handover_to_loaded={self.reward_weights['handover_to_loaded']}, "
                   f"handover_to_free={self.reward_weights['handover_to_free']}, "
                   f"rvt_weight={self.reward_weights['rvt_reward_weight']}")
        logger.info(f"  V9+ Dwell Time Constraint: {self.min_dwell_time_seconds}s "
                   f"({self.min_dwell_time_steps} steps) - Physical/Protocol constraint")

    def reset(self, seed=None, options=None):
        """Reset environment and establish episode-fixed candidate set"""
        super().reset(seed=seed)

        # V9.1: Smart resampling - ensure each episode has valid satellite coverage
        # Following academic standard: skip invalid episodes (no satellites)
        # Reference: MDPI Aerospace 2024 - "training ends when no available resource"
        MIN_REQUIRED_SATELLITES = 2
        MAX_RESAMPLE_ATTEMPTS = 100

        if options and 'start_time' in options:
            # Use specified time (for reproducible evaluation)
            self.episode_start = options['start_time']
            self.current_time = self.episode_start
            self._initialize_episode_candidates_internal()
        else:
            # Smart resampling: retry until we find a time with valid coverage
            for attempt in range(MAX_RESAMPLE_ATTEMPTS):
                random_minutes = np.random.randint(0, 30 * 24 * 60)  # 0-30 days
                self.episode_start = datetime(2025, 10, 26, 0, 0, 0) + timedelta(minutes=random_minutes)
                self.current_time = self.episode_start

                # Check satellite availability at this time
                self._initialize_episode_candidates_internal()
                available_count = sum(self.episode_candidate_available)

                if available_count >= MIN_REQUIRED_SATELLITES:
                    logger.debug(f"Found valid time after {attempt+1} attempts: {available_count} satellites")
                    break
            else:
                # Fallback to known-good time after all attempts fail
                logger.warning(f"Could not find valid time after {MAX_RESAMPLE_ATTEMPTS} attempts, using default")
                self.episode_start = datetime(2025, 10, 26, 0, 0, 0)
                self.current_time = self.episode_start
                self._initialize_episode_candidates_internal()

        self.previous_satellite = None
        self.handover_history = []

        # V9: Reset RVT cache and load tracking
        self.rvt_cache = {}
        self.satellite_load = {}

        # V9+: Reset dwell time counter
        self.steps_since_last_handover = 0

        # Reset statistics
        self.episode_stats = {
            'num_handovers': 0,
            'num_ping_pongs': 0,
            'avg_rsrp': 0.0,
            'timesteps': 0,
            'connectivity_ratio': 0.0,
            'connected_steps': 0,
            'avg_rvt': 0.0,
            'total_reward': 0.0,
        }

        # Log final candidate status (already initialized above)
        available_count = sum(self.episode_candidate_available)
        logger.info(f"Episode candidates initialized: {available_count}/{len(self.episode_candidate_ids)} available")

        # Get initial observation
        observation = self._get_observation()

        # Select initial satellite
        self._select_initial_satellite()

        # Generate action mask
        action_mask = self._get_action_mask()

        info = {
            'current_satellite': self.current_satellite,
            'num_visible': sum(self.episode_candidate_available),
            'num_candidates': len(self.episode_candidate_ids),
            'episode_start': self.episode_start.isoformat(),
            'current_time': self.current_time.isoformat(),
            'action_mask': action_mask,
            'candidate_ids': self.episode_candidate_ids.copy(),
        }

        logger.debug(f"Episode reset - {info['num_visible']}/{info['num_candidates']} candidates available")

        return observation, info

    def _initialize_episode_candidates(self):
        """Public wrapper for backward compatibility"""
        self._initialize_episode_candidates_internal()

    def _initialize_episode_candidates_internal(self):
        """Initialize episode-fixed candidate set (internal implementation)"""
        visible_satellites = []

        # Query all satellites at episode start
        for sat_id in self.satellite_ids:
            try:
                state_dict = self.adapter.calculate_state(
                    satellite_id=sat_id,
                    timestamp=self.current_time
                )

                if not state_dict:
                    continue

                elevation = state_dict.get('elevation_deg', -90)
                rsrp = state_dict.get('rsrp_dbm', -140)
                is_connectable = state_dict.get('is_connectable', False)

                # Include satellites visible or might rise soon
                if elevation > -30:
                    visible_satellites.append({
                        'id': sat_id,
                        'rsrp': rsrp,
                        'elevation': elevation,
                        'is_connectable': is_connectable,
                        'state_dict': state_dict,
                    })
            except Exception as e:
                logger.debug(f"Error querying {sat_id}: {e}")
                continue

        # Sort by elevation (highest first) for stability
        visible_satellites.sort(key=lambda x: (x['elevation'], x['rsrp']), reverse=True)

        # Select top-K as episode candidates
        top_k = visible_satellites[:self.max_visible_satellites]

        # Store FIXED candidate set for this episode
        self.episode_candidate_ids = [s['id'] for s in top_k]
        self.episode_candidate_states = [s['state_dict'] for s in top_k]
        self.episode_candidate_available = [
            s['elevation'] >= self.min_elevation_deg and s['is_connectable']
            for s in top_k
        ]

        # Pad with None if fewer than K candidates
        while len(self.episode_candidate_ids) < self.max_visible_satellites:
            self.episode_candidate_ids.append(None)
            self.episode_candidate_states.append(None)
            self.episode_candidate_available.append(False)

        available_count = sum(self.episode_candidate_available)
        logger.info(f"Episode candidates initialized: {available_count}/{len(self.episode_candidate_ids)} available")

    def _select_initial_satellite(self):
        """Select initial satellite from available candidates"""
        best_rsrp = -999
        best_idx = -1

        for i, (available, state) in enumerate(zip(
            self.episode_candidate_available,
            self.episode_candidate_states
        )):
            if available and state:
                rsrp = state.get('rsrp_dbm', -999)
                if rsrp > best_rsrp:
                    best_rsrp = rsrp
                    best_idx = i

        if best_idx >= 0:
            self.current_satellite = self.episode_candidate_ids[best_idx]
            self.handover_history.append(self.current_satellite)
            logger.debug(f"Initial satellite: {self.current_satellite} (RSRP: {best_rsrp:.1f} dBm)")
        else:
            self.current_satellite = None
            logger.warning("No available satellites at episode start")

    def step(self, action: int):
        """Execute one timestep with stable action semantics"""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        prev_satellite = self.current_satellite
        handover_occurred = False

        # V9+: Increment dwell time counter
        self.steps_since_last_handover += 1

        # Execute action
        if action == 0:
            handover_occurred = self._handle_stay_action()
        else:
            handover_occurred = self._handle_switch_action(action)

        # Advance time
        self.current_time += timedelta(seconds=self.time_step_seconds)

        # Update candidate states
        self._update_candidate_states()

        # Get observation
        observation = self._get_observation()

        # Calculate V9 reward
        reward = self._calculate_reward(
            handover_occurred=handover_occurred,
            prev_sat=prev_satellite,
        )

        # Check termination
        terminated, truncated = self._check_done()

        # Update statistics
        self._update_statistics(handover_occurred, reward)

        # Generate action mask
        action_mask = self._get_action_mask()

        info = {
            'current_satellite': self.current_satellite,
            'num_visible': sum(self.episode_candidate_available),
            'handover_occurred': handover_occurred,
            'current_time': self.current_time.isoformat(),
            'action_mask': action_mask,
            'episode_stats': self.episode_stats.copy(),
            **self.episode_stats,
        }

        return observation, reward, terminated, truncated, info

    def _handle_stay_action(self) -> bool:
        """Handle stay action (action=0)"""
        if self.current_satellite is None:
            return self._try_recover_satellite()

        # Check if current satellite still available
        if self.current_satellite in self.episode_candidate_ids:
            idx = self.episode_candidate_ids.index(self.current_satellite)
            if self.episode_candidate_available[idx]:
                return False  # Still connected, no handover

        # Current satellite lost - try to recover
        return self._try_recover_satellite()

    def _handle_switch_action(self, action: int) -> bool:
        """Handle switch action (action=1 to K)"""
        candidate_idx = action - 1
        target_satellite = self.episode_candidate_ids[candidate_idx]

        if target_satellite is None:
            return False

        if not self.episode_candidate_available[candidate_idx]:
            return False

        if target_satellite == self.current_satellite:
            return False

        # V9+: Minimum Dwell Time Constraint
        # Enforce physical/protocol constraint: cannot handover too frequently
        # Based on 3GPP NTN standards and academic research (typical: 30-60s)
        if self.current_satellite is not None:  # Not the first handover
            if self.steps_since_last_handover < self.min_dwell_time_steps:
                # Dwell time not met - force stay on current satellite
                logger.debug(f"Handover blocked by dwell time constraint: "
                           f"{self.steps_since_last_handover}/{self.min_dwell_time_steps} steps "
                           f"({self.steps_since_last_handover * self.time_step_seconds}/"
                           f"{self.min_dwell_time_seconds}s)")
                return False  # Stay on current satellite

        # Execute handover
        self.previous_satellite = self.current_satellite
        self.current_satellite = target_satellite
        self.handover_history.append(target_satellite)

        # V9+: Reset dwell time counter on successful handover
        self.steps_since_last_handover = 0

        if len(self.handover_history) > 10:
            self.handover_history = self.handover_history[-10:]

        logger.debug(f"Handover: {self.previous_satellite} -> {self.current_satellite}")
        return True

    def _try_recover_satellite(self) -> bool:
        """Try to recover connection to best available satellite"""
        best_rsrp = -999
        best_idx = -1

        for i, (available, state) in enumerate(zip(
            self.episode_candidate_available,
            self.episode_candidate_states
        )):
            if available and state:
                rsrp = state.get('rsrp_dbm', -999)
                if rsrp > best_rsrp:
                    best_rsrp = rsrp
                    best_idx = i

        if best_idx >= 0:
            new_satellite = self.episode_candidate_ids[best_idx]
            if new_satellite != self.current_satellite:
                self.previous_satellite = self.current_satellite
                self.current_satellite = new_satellite
                self.handover_history.append(new_satellite)

                # V9+: Reset dwell time counter on recovery handover
                self.steps_since_last_handover = 0

                logger.debug(f"Recovery handover: {self.previous_satellite} -> {new_satellite}")
                return True
        else:
            self.current_satellite = None

        return False

    def _update_candidate_states(self):
        """Update states for episode candidates"""
        for i, sat_id in enumerate(self.episode_candidate_ids):
            if sat_id is None:
                continue

            try:
                state_dict = self.adapter.calculate_state(
                    satellite_id=sat_id,
                    timestamp=self.current_time
                )

                if state_dict:
                    self.episode_candidate_states[i] = state_dict
                    elevation = state_dict.get('elevation_deg', -90)
                    is_connectable = state_dict.get('is_connectable', False)
                    self.episode_candidate_available[i] = (
                        elevation >= self.min_elevation_deg and is_connectable
                    )
                else:
                    self.episode_candidate_available[i] = False

            except Exception as e:
                logger.debug(f"Error updating {sat_id}: {e}")
                self.episode_candidate_available[i] = False

    def _get_observation(self) -> np.ndarray:
        """
        Generate observation with FIXED candidate order.
        V9: Added RVT dimension (14 total dimensions).
        """
        observation = np.zeros(
            (self.max_visible_satellites, 14),  # V9: 13 → 14 dimensions (added RVT)
            dtype=np.float32
        )

        for i, (sat_id, state, available) in enumerate(zip(
            self.episode_candidate_ids,
            self.episode_candidate_states,
            self.episode_candidate_available
        )):
            if sat_id and state:
                is_current = (sat_id == self.current_satellite)

                # V9: Calculate RVT for this satellite
                rvt_seconds = self._calculate_rvt(sat_id)

                observation[i] = self._state_dict_to_vector(
                    state,
                    is_current=is_current,
                    rvt_seconds=rvt_seconds
                )

                if not available:
                    observation[i, 4] = -90.0  # Mark unavailable

        # NaN/Inf check
        if np.isnan(observation).any() or np.isinf(observation).any():
            observation = np.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6)

        return observation

    def _state_dict_to_vector(self, state_dict: Dict, is_current: bool = False,
                             rvt_seconds: float = 0.0) -> np.ndarray:
        """
        Convert state dict to 14-dimensional vector.
        V9: Added RVT as 14th dimension.
        """
        return np.array([
            state_dict.get('rsrp_dbm', -140),
            state_dict.get('rsrq_db', -20),
            state_dict.get('rs_sinr_db', -10),
            state_dict.get('distance_km', 1000),
            state_dict.get('elevation_deg', 0),
            state_dict.get('doppler_shift_hz', 0),
            state_dict.get('path_loss_db', 150),
            state_dict.get('atmospheric_loss_db', 1),
            state_dict.get('radial_velocity_ms', 0),
            state_dict.get('offset_mo_db', 0),
            state_dict.get('cell_offset_db', 0),
            state_dict.get('propagation_delay_ms', 5),
            10.0 if is_current else 0.0,
            rvt_seconds / 60.0,  # V9: RVT in minutes (normalized)
        ], dtype=np.float32)

    def _calculate_rvt(self, sat_id: str) -> float:
        """
        Calculate Remaining Visibility Time (RVT) for a satellite.

        Returns:
            RVT in seconds (time until satellite drops below min_elevation)
        """
        # Check cache
        if sat_id in self.rvt_cache:
            cache_time, cached_rvt = self.rvt_cache[sat_id]
            if (self.current_time - cache_time).total_seconds() < self.rvt_cache_duration:
                return cached_rvt

        # Calculate RVT by looking ahead
        rvt_seconds = 0.0
        check_time = self.current_time
        max_lookahead = timedelta(minutes=self.rvt_lookahead_minutes)

        try:
            while (check_time - self.current_time) < max_lookahead:
                check_time += timedelta(seconds=self.rvt_check_interval_seconds)

                state_dict = self.adapter.calculate_state(
                    satellite_id=sat_id,
                    timestamp=check_time
                )

                if not state_dict:
                    break

                elevation = state_dict.get('elevation_deg', -90)

                if elevation < self.min_elevation_deg:
                    # Found horizon crossing
                    rvt_seconds = (check_time - self.current_time).total_seconds()
                    break
            else:
                # Satellite stays visible for entire lookahead period
                rvt_seconds = max_lookahead.total_seconds()

        except Exception as e:
            logger.debug(f"Error calculating RVT for {sat_id}: {e}")
            rvt_seconds = 0.0

        # Cache result
        self.rvt_cache[sat_id] = (self.current_time, rvt_seconds)

        return rvt_seconds

    def _get_satellite_load(self, sat_id: str) -> float:
        """
        Get load factor for a satellite (simplified model).

        In a real system, this would query satellite capacity/user count.
        Here we use a simplified model based on recent handover activity.

        Returns:
            Load factor 0.0 to 1.0
        """
        # Simplified: Random load with slight persistence
        if sat_id not in self.satellite_load:
            self.satellite_load[sat_id] = np.random.uniform(0.2, 0.8)

        # Add some noise for variation
        self.satellite_load[sat_id] = np.clip(
            self.satellite_load[sat_id] + np.random.normal(0, 0.05),
            0.0, 1.0
        )

        return self.satellite_load[sat_id]

    def _is_satellite_loaded(self, sat_id: str) -> bool:
        """Check if satellite is considered loaded/overloaded"""
        load = self._get_satellite_load(sat_id)
        return load > self.load_threshold

    def _calculate_reward(self, handover_occurred: bool, prev_sat: str) -> float:
        """
        V9: RVT-based reward function (IEEE TAES 2024 Equation 14)

        Reward structure:
        - If handover to loaded satellite: -500 (z1)
        - If handover to free satellite: -300 (z2)
        - If stay on loaded satellite: -100 * load_factor (f1)
        - If stay on free satellite: +RVT (f2)

        Additional components:
        - Small QoS bonus (minimal, not primary)
        - Ping-pong penalty
        """
        reward = 0.0

        # No connection - heavy penalty
        if self.current_satellite is None:
            return -10.0

        # Get current satellite load and RVT
        is_loaded = self._is_satellite_loaded(self.current_satellite)
        rvt_seconds = self._calculate_rvt(self.current_satellite)

        if handover_occurred:
            # HANDOVER REWARD (from paper)
            if is_loaded:
                # z1: Handover to loaded/overloaded satellite
                reward += self.reward_weights['handover_to_loaded']  # -500
            else:
                # z2: Handover to available satellite
                reward += self.reward_weights['handover_to_free']  # -300

            # Check for ping-pong pattern
            if len(self.handover_history) >= 3:
                recent = self.handover_history[-3:]
                if len(set(recent)) < len(recent):
                    reward += self.reward_weights['ping_pong_penalty']  # -50
                    self.episode_stats['num_ping_pongs'] += 1

        else:
            # STAY REWARD (from paper)
            if is_loaded:
                # f1: Stay on loaded satellite (negative reward)
                load_factor = self._get_satellite_load(self.current_satellite)
                reward -= self.reward_weights['stay_loaded_factor'] * load_factor  # -100 * load
            else:
                # f2: Stay on free satellite (RVT reward)
                reward += self.reward_weights['rvt_reward_weight'] * rvt_seconds  # +RVT

        # Add small QoS component (minimal, not primary)
        if self.current_satellite in self.episode_candidate_ids:
            idx = self.episode_candidate_ids.index(self.current_satellite)
            state = self.episode_candidate_states[idx]

            if state:
                rsrp = state.get('rsrp_dbm', -140)
                rsrp_normalized = np.clip((rsrp + 110) / 50, 0, 1)
                reward += rsrp_normalized * self.reward_weights['qos']  # Small QoS bonus

        return float(reward)

    def _check_done(self) -> Tuple[bool, bool]:
        """Check if episode is done"""
        episode_duration = timedelta(minutes=self.episode_duration_minutes)
        if self.current_time >= self.episode_start + episode_duration:
            return False, True
        return False, False

    def _update_statistics(self, handover_occurred: bool, reward: float):
        """Update episode statistics"""
        self.episode_stats['timesteps'] += 1
        self.episode_stats['total_reward'] += reward

        if handover_occurred:
            self.episode_stats['num_handovers'] += 1

        if self.current_satellite is not None:
            self.episode_stats['connected_steps'] += 1

            # Update average RSRP
            if self.current_satellite in self.episode_candidate_ids:
                idx = self.episode_candidate_ids.index(self.current_satellite)
                state = self.episode_candidate_states[idx]
                if state:
                    rsrp = state.get('rsrp_dbm', 0)
                    n = self.episode_stats['timesteps']
                    old_avg = self.episode_stats['avg_rsrp']
                    self.episode_stats['avg_rsrp'] = old_avg + (rsrp - old_avg) / n

                    # V9: Update average RVT
                    rvt = self._calculate_rvt(self.current_satellite)
                    old_avg_rvt = self.episode_stats['avg_rvt']
                    self.episode_stats['avg_rvt'] = old_avg_rvt + (rvt - old_avg_rvt) / n

        # Update connectivity ratio
        self.episode_stats['connectivity_ratio'] = (
            self.episode_stats['connected_steps'] / self.episode_stats['timesteps']
        )

    def _get_action_mask(self) -> np.ndarray:
        """Generate action mask for valid actions"""
        action_mask = np.zeros(self.action_space.n, dtype=bool)

        # Action 0 (stay) always valid
        action_mask[0] = True

        # Actions 1-K valid if candidate is available AND not current satellite
        for i, (sat_id, available) in enumerate(zip(
            self.episode_candidate_ids,
            self.episode_candidate_available
        )):
            if available and sat_id != self.current_satellite:
                action_mask[i + 1] = True

        return action_mask

    def render(self):
        pass

    def close(self):
        pass
