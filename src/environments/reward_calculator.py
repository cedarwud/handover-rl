#!/usr/bin/env python3
"""
RVT-Based Reward Calculator

從 satellite_handover_env.py 抽取的獎勵計算模組。

Reference: "User-Centric Satellite Handover for Multiple Traffic Profiles
Using Deep Q-Learning" (IEEE TAES 2024, Equation 14)

Reward Structure:
- Handover to loaded satellite: -500 (z1)
- Handover to free satellite: -300 (z2)
- Stay on loaded satellite: -100 * load_factor (f1)
- Stay on free satellite: +RVT (f2, Remaining Visibility Time in seconds)

Key Innovation: RVT (Remaining Visibility Time)
- Predicts when satellite will drop below min_elevation
- Encourages switching before satellite loss
- Natural incentive for proactive handovers
"""

import logging
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RVTRewardCalculator:
    """RVT-based reward function (IEEE TAES 2024 Equation 14)"""

    def __init__(self, reward_weights: Dict, load_threshold: float = 0.7):
        self.reward_weights = reward_weights
        self.load_threshold = load_threshold

        # Satellite load tracking (simplified: based on recent handover rate)
        self.satellite_load = {}  # sat_id -> load_factor (0.0 to 1.0)

        # RVT cache
        self.rvt_cache = {}  # sat_id -> (timestamp, rvt_seconds)
        self.rvt_cache_duration = 60  # Cache valid for 60 seconds

        # RVT calculation parameters
        self.rvt_lookahead_minutes = 60
        self.rvt_check_interval_seconds = 30

    def reset(self):
        """Reset state for new episode"""
        self.rvt_cache = {}
        self.satellite_load = {}

    def calculate_reward(
        self,
        handover_occurred: bool,
        current_satellite: Optional[str],
        handover_history: List[str],
        episode_candidate_ids: List[Optional[str]],
        episode_candidate_states: List[Optional[Dict]],
        adapter,
        current_time,
        min_elevation_deg: float,
    ) -> float:
        """
        RVT-based reward function (IEEE TAES 2024 Equation 14)

        Args:
            handover_occurred: Whether a handover just happened
            current_satellite: Current serving satellite ID
            handover_history: Recent handover history
            episode_candidate_ids: Fixed candidate satellite IDs
            episode_candidate_states: Current state dicts for candidates
            adapter: OrbitEngineAdapter for RVT calculation
            current_time: Current simulation time
            min_elevation_deg: Minimum elevation threshold

        Returns:
            Reward value
        """
        # No connection - heavy penalty
        if current_satellite is None:
            return -10.0

        reward = 0.0

        # Get current satellite load and RVT
        is_loaded = self._is_satellite_loaded(current_satellite)
        rvt_seconds = self.calculate_rvt(
            current_satellite, adapter, current_time, min_elevation_deg
        )

        if handover_occurred:
            # HANDOVER REWARD (from paper)
            if is_loaded:
                # z1: Handover to loaded/overloaded satellite
                reward += self.reward_weights['handover_to_loaded']  # -500
            else:
                # z2: Handover to available satellite
                reward += self.reward_weights['handover_to_free']  # -300

            # Check for ping-pong pattern
            if len(handover_history) >= 3:
                recent = handover_history[-3:]
                if len(set(recent)) < len(recent):
                    reward += self.reward_weights['ping_pong_penalty']  # -50

        else:
            # STAY REWARD (from paper)
            if is_loaded:
                # f1: Stay on loaded satellite (negative reward)
                load_factor = self._get_satellite_load(current_satellite)
                reward -= self.reward_weights['stay_loaded_factor'] * load_factor
            else:
                # f2: Stay on free satellite (RVT reward)
                reward += self.reward_weights['rvt_reward_weight'] * rvt_seconds

        # Add small QoS component (minimal, not primary)
        if current_satellite in episode_candidate_ids:
            idx = episode_candidate_ids.index(current_satellite)
            state = episode_candidate_states[idx]

            if state:
                rsrp = state.get('rsrp_dbm', -140)
                rsrp_normalized = np.clip((rsrp + 110) / 50, 0, 1)
                reward += rsrp_normalized * self.reward_weights['qos']

        return float(reward)

    def calculate_rvt(
        self,
        sat_id: str,
        adapter,
        current_time,
        min_elevation_deg: float,
    ) -> float:
        """
        Calculate Remaining Visibility Time (RVT) for a satellite.

        Returns:
            RVT in seconds (time until satellite drops below min_elevation)
        """
        # Check cache
        if sat_id in self.rvt_cache:
            cache_time, cached_rvt = self.rvt_cache[sat_id]
            if (current_time - cache_time).total_seconds() < self.rvt_cache_duration:
                return cached_rvt

        # Calculate RVT by looking ahead
        rvt_seconds = 0.0
        check_time = current_time
        max_lookahead = timedelta(minutes=self.rvt_lookahead_minutes)

        try:
            while (check_time - current_time) < max_lookahead:
                check_time += timedelta(seconds=self.rvt_check_interval_seconds)

                state_dict = adapter.calculate_state(
                    satellite_id=sat_id,
                    timestamp=check_time
                )

                if not state_dict:
                    break

                elevation = state_dict.get('elevation_deg', -90)

                if elevation < min_elevation_deg:
                    rvt_seconds = (check_time - current_time).total_seconds()
                    break
            else:
                # Satellite stays visible for entire lookahead period
                rvt_seconds = max_lookahead.total_seconds()

        except Exception as e:
            logger.debug(f"Error calculating RVT for {sat_id}: {e}")
            rvt_seconds = 0.0

        # Cache result
        self.rvt_cache[sat_id] = (current_time, rvt_seconds)

        return rvt_seconds

    def _get_satellite_load(self, sat_id: str) -> float:
        """
        Get load factor for a satellite (simplified model).

        Returns:
            Load factor 0.0 to 1.0
        """
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
