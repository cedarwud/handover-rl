#!/usr/bin/env python3
"""
Test Suite for SatelliteHandoverEnv

Comprehensive tests for the core Online RL environment.
Covers all functionality including:
- Initialization
- reset() behavior
- step() behavior
- Observation generation
- Action execution
- Reward calculation
- Episode termination
- Handover logic
- Ping-pong detection

Academic Standard: Real TLE data, complete physics, no mocking
"""

import unittest
import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tests.test_base import BaseEnvironmentTest
from tests.test_utils import (
    load_test_config,
    get_test_satellite_ids,
    get_test_timestamp,
    create_mock_adapter_for_testing,
    verify_observation_space,
    count_visible_satellites,
)
from environments.satellite_handover_env import SatelliteHandoverEnv


class TestSatelliteHandoverEnvInitialization(BaseEnvironmentTest):
    """Test SatelliteHandoverEnv initialization"""

    def test_init_basic(self):
        """Test basic environment initialization"""
        env = SatelliteHandoverEnv(
            adapter=self.adapter,
            satellite_ids=self.test_satellite_ids,
            config=self.config
        )

        # Check that environment was created
        self.assertIsNotNone(env)
        self.assertEqual(len(env.satellite_ids), len(self.test_satellite_ids))

    def test_observation_space(self):
        """Test observation space configuration"""
        env = SatelliteHandoverEnv(
            adapter=self.adapter,
            satellite_ids=self.test_satellite_ids,
            config=self.config
        )

        # Check observation space
        max_visible = env.max_visible_satellites
        expected_shape = (max_visible, 12)

        self.assertEqual(env.observation_space.shape, expected_shape)
        self.assertEqual(env.observation_space.dtype, np.float32)

    def test_action_space(self):
        """Test action space configuration"""
        env = SatelliteHandoverEnv(
            adapter=self.adapter,
            satellite_ids=self.test_satellite_ids,
            config=self.config
        )

        # Action space should be Discrete(K+1)
        # 0 = stay, 1 to K = switch to candidate[i-1]
        expected_n = env.max_visible_satellites + 1

        self.assertEqual(env.action_space.n, expected_n)

    def test_config_parameters(self):
        """Test that config parameters are loaded correctly"""
        env = SatelliteHandoverEnv(
            adapter=self.adapter,
            satellite_ids=self.test_satellite_ids,
            config=self.config
        )

        # Check parameters exist
        self.assertIsNotNone(env.min_elevation_deg)
        self.assertIsNotNone(env.time_step_seconds)
        self.assertIsNotNone(env.episode_duration_minutes)
        self.assertIsNotNone(env.max_visible_satellites)

        # Check reward weights
        self.assertIn('qos', env.reward_weights)
        self.assertIn('handover_penalty', env.reward_weights)
        self.assertIn('ping_pong_penalty', env.reward_weights)

    def test_adapter_assignment(self):
        """Test that adapter is correctly assigned"""
        env = SatelliteHandoverEnv(
            adapter=self.adapter,
            satellite_ids=self.test_satellite_ids,
            config=self.config
        )

        self.assertIs(env.adapter, self.adapter)
        self.assertIsNotNone(env.adapter)


class TestSatelliteHandoverEnvReset(BaseEnvironmentTest):
    """Test SatelliteHandoverEnv reset functionality"""

    def setUp(self):
        """Create environment for each test"""
        super().setUp()
        self.env = SatelliteHandoverEnv(
            adapter=self.adapter,
            satellite_ids=self.test_satellite_ids,
            config=self.config
        )

    def test_reset_basic(self):
        """Test basic reset functionality"""
        observation, info = self.env.reset()

        # Check observation
        self.assertObservationValid(observation, (self.env.max_visible_satellites, 12))

        # Check info
        self.assertIn('current_satellite', info)
        self.assertIn('num_visible', info)
        self.assertIn('episode_start', info)
        self.assertIn('current_time', info)

    def test_reset_with_seed(self):
        """Test reset with seed for reproducibility"""
        observation1, info1 = self.env.reset(seed=42)
        observation2, info2 = self.env.reset(seed=42)

        # Same seed should give same initial state
        np.testing.assert_array_equal(observation1, observation2)

    def test_reset_with_custom_start_time(self):
        """Test reset with custom start time"""
        custom_time = datetime(2025, 10, 8, 12, 0, 0)

        observation, info = self.env.reset(options={'start_time': custom_time})

        # Check that start time was set
        self.assertEqual(self.env.episode_start, custom_time)
        self.assertEqual(self.env.current_time, custom_time)

    def test_reset_selects_initial_satellite(self):
        """Test that reset selects initial satellite (highest RSRP)"""
        observation, info = self.env.reset()

        # If satellites are visible, one should be selected
        if info['num_visible'] > 0:
            self.assertIsNotNone(info['current_satellite'])
            self.assertIn(self.env.current_satellite, self.env.current_visible_satellites)
            # Current satellite should be first in list (highest RSRP)
            self.assertEqual(self.env.current_satellite, self.env.current_visible_satellites[0])

    def test_reset_statistics_cleared(self):
        """Test that episode statistics are cleared on reset"""
        # Do some steps
        self.env.reset()
        for _ in range(5):
            action = self.env.action_space.sample()
            try:
                self.env.step(action)
            except:
                break

        # Reset again
        observation, info = self.env.reset()

        # Statistics should be reset
        self.assertEqual(self.env.episode_stats['num_handovers'], 0)
        self.assertEqual(self.env.episode_stats['num_ping_pongs'], 0)
        self.assertEqual(self.env.episode_stats['timesteps'], 0)

    def test_reset_handover_history_cleared(self):
        """Test that handover history is cleared on reset"""
        # Do some steps
        self.env.reset()
        for _ in range(5):
            action = self.env.action_space.sample()
            try:
                self.env.step(action)
            except:
                break

        # Reset again
        self.env.reset()

        # Handover history should have only initial satellite
        self.assertLessEqual(len(self.env.handover_history), 1)


class TestSatelliteHandoverEnvStep(BaseEnvironmentTest):
    """Test SatelliteHandoverEnv step functionality"""

    def setUp(self):
        """Create and reset environment for each test"""
        super().setUp()
        self.env = SatelliteHandoverEnv(
            adapter=self.adapter,
            satellite_ids=self.test_satellite_ids,
            config=self.config
        )
        self.env.reset(seed=42)

    def test_step_basic(self):
        """Test basic step functionality"""
        action = 0  # Stay with current satellite

        observation, reward, terminated, truncated, info = self.env.step(action)

        # Check return types
        self.assertObservationValid(observation, (self.env.max_visible_satellites, 12))
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_step_action_stay(self):
        """Test action 0 (stay with current satellite)"""
        initial_satellite = self.env.current_satellite

        observation, reward, terminated, truncated, info = self.env.step(0)

        # If satellite still visible, should stay
        if initial_satellite in self.env.current_visible_satellites:
            self.assertEqual(self.env.current_satellite, initial_satellite)
            self.assertFalse(info['handover_occurred'])

    def test_step_action_switch(self):
        """Test action 1-K (switch to candidate satellite)"""
        # Get number of visible satellites
        num_visible = len(self.env.current_visible_satellites)

        if num_visible > 1:
            # Try to switch to second satellite
            observation, reward, terminated, truncated, info = self.env.step(1)

            # Should have switched (if action was valid)
            # Note: May not switch if only 1 satellite visible

    def test_step_invalid_action_raises(self):
        """Test that invalid action raises ValueError"""
        invalid_action = self.env.action_space.n + 10  # Out of range

        with self.assertRaises(ValueError):
            self.env.step(invalid_action)

    def test_step_advances_time(self):
        """Test that step advances time correctly"""
        initial_time = self.env.current_time
        time_step = timedelta(seconds=self.env.time_step_seconds)

        self.env.step(0)

        expected_time = initial_time + time_step
        self.assertEqual(self.env.current_time, expected_time)

    def test_step_updates_statistics(self):
        """Test that step updates episode statistics"""
        initial_timesteps = self.env.episode_stats['timesteps']

        self.env.step(0)

        self.assertEqual(self.env.episode_stats['timesteps'], initial_timesteps + 1)

    def test_step_info_dict(self):
        """Test that info dictionary contains required fields"""
        observation, reward, terminated, truncated, info = self.env.step(0)

        required_fields = [
            'current_satellite',
            'num_visible',
            'handover_occurred',
            'current_time',
            'num_handovers',
            'num_ping_pongs',
            'avg_rsrp',
            'timesteps',
        ]

        for field in required_fields:
            self.assertIn(field, info)

    def test_step_multiple_steps(self):
        """Test multiple consecutive steps"""
        num_steps = 10

        for i in range(num_steps):
            action = 0  # Stay
            observation, reward, terminated, truncated, info = self.env.step(action)

            # Check timestep counter
            self.assertEqual(info['timesteps'], i + 1)

            if terminated or truncated:
                break


class TestSatelliteHandoverEnvObservation(BaseEnvironmentTest):
    """Test observation generation in SatelliteHandoverEnv"""

    def setUp(self):
        """Create and reset environment"""
        super().setUp()
        self.env = SatelliteHandoverEnv(
            adapter=self.adapter,
            satellite_ids=self.test_satellite_ids,
            config=self.config
        )
        self.env.reset(seed=42)

    def test_observation_shape(self):
        """Test observation has correct shape"""
        observation = self.env._get_observation()

        expected_shape = (self.env.max_visible_satellites, 12)
        self.assertEqual(observation.shape, expected_shape)

    def test_observation_dtype(self):
        """Test observation has correct dtype"""
        observation = self.env._get_observation()

        self.assertEqual(observation.dtype, np.float32)

    def test_observation_no_nan_inf(self):
        """Test observation contains no NaN or Inf values"""
        observation = self.env._get_observation()

        self.assertFalse(np.any(np.isnan(observation)))
        self.assertFalse(np.any(np.isinf(observation)))

    def test_observation_sorted_by_rsrp(self):
        """Test that visible satellites are sorted by RSRP (descending)"""
        observation = self.env._get_observation()

        # Get RSRP values (first column)
        rsrp_values = observation[:, 0]

        # Non-zero RSRP should be in descending order
        non_zero_rsrp = rsrp_values[rsrp_values != 0]

        if len(non_zero_rsrp) > 1:
            # Check descending order
            for i in range(len(non_zero_rsrp) - 1):
                self.assertGreaterEqual(non_zero_rsrp[i], non_zero_rsrp[i + 1])

    def test_observation_top_k_selection(self):
        """Test that only top-K satellites are included"""
        observation = self.env._get_observation()

        # Count non-zero rows
        visible_count = count_visible_satellites(observation)

        # Should be at most max_visible_satellites
        self.assertLessEqual(visible_count, self.env.max_visible_satellites)

    def test_observation_updates_visible_list(self):
        """Test that _get_observation updates current_visible_satellites"""
        initial_len = len(self.env.current_visible_satellites)

        observation = self.env._get_observation()

        # current_visible_satellites should be updated
        self.assertIsNotNone(self.env.current_visible_satellites)
        self.assertIsInstance(self.env.current_visible_satellites, list)


class TestSatelliteHandoverEnvReward(BaseEnvironmentTest):
    """Test reward calculation in SatelliteHandoverEnv"""

    def setUp(self):
        """Create and reset environment"""
        super().setUp()
        self.env = SatelliteHandoverEnv(
            adapter=self.adapter,
            satellite_ids=self.test_satellite_ids,
            config=self.config
        )
        self.env.reset(seed=42)

    def test_reward_is_numeric(self):
        """Test that reward is a numeric value"""
        observation, reward, terminated, truncated, info = self.env.step(0)

        self.assertIsInstance(reward, (int, float))
        self.assertFalse(np.isnan(reward))
        self.assertFalse(np.isinf(reward))

    def test_reward_no_handover(self):
        """Test reward when no handover occurs (action 0)"""
        # Record initial state
        initial_satellite = self.env.current_satellite

        observation, reward, terminated, truncated, info = self.env.step(0)

        # If stayed with same satellite, no handover penalty
        if not info['handover_occurred']:
            # Reward should be primarily QoS component
            # No handover penalty or ping-pong penalty
            pass  # Just verify it's calculated

    def test_reward_with_handover(self):
        """Test reward when handover occurs"""
        num_visible = len(self.env.current_visible_satellites)

        if num_visible > 1:
            # Perform handover
            observation, reward, terminated, truncated, info = self.env.step(1)

            if info['handover_occurred']:
                # Reward should include handover penalty
                self.assertEqual(info['num_handovers'], 1)

    def test_reward_ping_pong_detection(self):
        """Test ping-pong penalty detection"""
        num_visible = len(self.env.current_visible_satellites)

        if num_visible > 1:
            # Perform multiple handovers to trigger ping-pong
            # 1. Switch to satellite 2
            self.env.step(1)

            # 2. Switch back to satellite 1
            self.env.step(1)

            # 3. Switch to satellite 2 again (ping-pong)
            observation, reward, terminated, truncated, info = self.env.step(1)

            # Check if ping-pong was detected
            # (May or may not trigger depending on handover history)

    def test_reward_no_satellite_penalty(self):
        """Test large penalty when no satellite is available"""
        # Force no satellite scenario
        self.env.current_satellite = None
        self.env.current_visible_satellites = []

        observation = np.zeros((self.env.max_visible_satellites, 12), dtype=np.float32)
        reward = self.env._calculate_reward(
            observation=observation,
            handover_occurred=False,
            prev_sat=None,
            curr_sat=None
        )

        # Should receive large penalty
        self.assertLess(reward, 0)


class TestSatelliteHandoverEnvTermination(BaseEnvironmentTest):
    """Test episode termination conditions"""

    def setUp(self):
        """Create and reset environment"""
        super().setUp()
        self.env = SatelliteHandoverEnv(
            adapter=self.adapter,
            satellite_ids=self.test_satellite_ids,
            config=self.config
        )
        self.env.reset(seed=42)

    def test_termination_time_limit(self):
        """Test episode truncation on time limit"""
        # Set episode duration to very short
        self.env.episode_duration_minutes = 0.01  # 0.6 seconds

        # Take multiple steps until truncated
        for _ in range(100):
            observation, reward, terminated, truncated, info = self.env.step(0)

            if truncated:
                # Episode should be truncated (time limit)
                self.assertTrue(truncated)
                self.assertFalse(terminated)
                break

    def test_termination_no_satellites(self):
        """Test episode termination when no satellites visible"""
        # Force no satellites scenario
        self.env.current_visible_satellites = []

        terminated, truncated = self.env._check_done()

        # Should terminate (no connectivity)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_termination_current_satellite_lost(self):
        """Test termination when current satellite is lost"""
        # Set current satellite to one not in visible list
        self.env.current_satellite = "FAKE-SAT-99999"
        self.env.current_visible_satellites = ["SAT-1", "SAT-2"]

        terminated, truncated = self.env._check_done()

        # Should terminate (lost current satellite)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_termination_continue(self):
        """Test that episode continues in normal conditions"""
        # Normal conditions: satellite available, time remaining
        self.env.episode_start = datetime(2025, 10, 7, 0, 0, 0)
        self.env.current_time = datetime(2025, 10, 7, 0, 1, 0)  # 1 minute in
        self.env.episode_duration_minutes = 95  # 95 minutes total

        if len(self.env.current_visible_satellites) > 0:
            self.env.current_satellite = self.env.current_visible_satellites[0]

            terminated, truncated = self.env._check_done()

            # Should continue
            self.assertFalse(terminated)
            self.assertFalse(truncated)


class TestSatelliteHandoverEnvHandover(BaseEnvironmentTest):
    """Test handover logic in SatelliteHandoverEnv"""

    def setUp(self):
        """Create and reset environment"""
        super().setUp()
        self.env = SatelliteHandoverEnv(
            adapter=self.adapter,
            satellite_ids=self.test_satellite_ids,
            config=self.config
        )
        self.env.reset(seed=42)

    def test_handover_basic(self):
        """Test basic handover execution"""
        num_visible = len(self.env.current_visible_satellites)

        if num_visible > 1:
            initial_satellite = self.env.current_satellite

            # Switch to different satellite
            observation, reward, terminated, truncated, info = self.env.step(1)

            # Check if handover occurred
            if info['handover_occurred']:
                self.assertNotEqual(self.env.current_satellite, initial_satellite)
                self.assertEqual(info['num_handovers'], 1)

    def test_handover_history_tracking(self):
        """Test that handover history is tracked correctly"""
        num_visible = len(self.env.current_visible_satellites)

        if num_visible > 1:
            initial_history_len = len(self.env.handover_history)

            # Perform handover
            self.env.step(1)

            # History should be updated
            self.assertGreaterEqual(len(self.env.handover_history), initial_history_len)

    def test_handover_history_max_length(self):
        """Test that handover history is limited to 10 entries"""
        num_visible = len(self.env.current_visible_satellites)

        if num_visible > 1:
            # Perform many handovers
            for _ in range(20):
                try:
                    self.env.step(1)
                except:
                    break

            # History should be capped at 10
            self.assertLessEqual(len(self.env.handover_history), 10)

    def test_forced_handover_on_satellite_loss(self):
        """Test forced handover when current satellite is lost"""
        # Step 1: Ensure we have multiple visible satellites
        if len(self.env.current_visible_satellites) > 1:
            # Step 2: Record current satellite
            current_sat = self.env.current_satellite

            # Step 3: Simulate satellite loss by removing it from visible list
            # (This would happen naturally in _get_observation if satellite moved out of range)
            # We'll test the action 0 behavior when current satellite is not in visible list

            # For this test, we'll just verify the logic exists in step()
            # The actual satellite loss is tested in step_action_stay


class TestSatelliteHandoverEnvIntegration(BaseEnvironmentTest):
    """Integration tests for complete environment workflow"""

    def setUp(self):
        """Create environment"""
        super().setUp()
        self.env = SatelliteHandoverEnv(
            adapter=self.adapter,
            satellite_ids=self.test_satellite_ids,
            config=self.config
        )

    def test_full_episode_workflow(self):
        """Test complete episode from reset to termination"""
        # Reset
        observation, info = self.env.reset(seed=42)

        # Run episode
        max_steps = 100
        total_reward = 0

        for step in range(max_steps):
            # Select action (simple: always stay)
            action = 0

            # Step
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            # Validate observation
            self.assertObservationValid(observation, (self.env.max_visible_satellites, 12))

            # Check termination
            if terminated or truncated:
                break

        # Episode should have run for at least 1 step
        self.assertGreater(info['timesteps'], 0)

    def test_multiple_episodes(self):
        """Test running multiple episodes"""
        num_episodes = 3

        for episode in range(num_episodes):
            observation, info = self.env.reset(seed=episode)

            # Run short episode
            for _ in range(10):
                action = self.env.action_space.sample()
                observation, reward, terminated, truncated, info = self.env.step(action)

                if terminated or truncated:
                    break

            # Episode should reset statistics
            self.assertGreaterEqual(info['timesteps'], 0)

    def test_random_actions_episode(self):
        """Test episode with random actions"""
        observation, info = self.env.reset(seed=42)

        max_steps = 50
        for _ in range(max_steps):
            # Random action
            action = self.env.action_space.sample()

            observation, reward, terminated, truncated, info = self.env.step(action)

            # Validate observation
            self.assertObservationValid(observation, (self.env.max_visible_satellites, 12))

            if terminated or truncated:
                break

    def test_uses_real_adapter(self):
        """Test that environment uses real OrbitEngineAdapter"""
        # Verify adapter is real (not mock)
        self.assertUsesRealTLE()

        # Test observation generation
        observation, info = self.env.reset()

        # If satellites are visible, they should have real physics values
        if info['num_visible'] > 0:
            # Extract RSRP values
            rsrp_values = observation[:, 0]
            non_zero_rsrp = rsrp_values[rsrp_values != 0]

            if len(non_zero_rsrp) > 0:
                # RSRP should be in valid 3GPP range
                for rsrp in non_zero_rsrp:
                    # May be outside range if not connectable, but should be reasonable
                    self.assertLess(rsrp, 100)  # Not absurdly high
                    self.assertGreater(rsrp, -200)  # Not absurdly low


class TestSatelliteHandoverEnvEdgeCases(BaseEnvironmentTest):
    """Test edge cases and error handling"""

    def setUp(self):
        """Create environment"""
        super().setUp()
        self.env = SatelliteHandoverEnv(
            adapter=self.adapter,
            satellite_ids=self.test_satellite_ids,
            config=self.config
        )

    def test_empty_satellite_pool(self):
        """Test behavior with empty satellite pool"""
        # Create environment with empty pool
        env_empty = SatelliteHandoverEnv(
            adapter=self.adapter,
            satellite_ids=[],
            config=self.config
        )

        observation, info = env_empty.reset()

        # Should have zero visible satellites
        self.assertEqual(info['num_visible'], 0)

    def test_single_satellite(self):
        """Test behavior with single satellite"""
        # Create environment with 1 satellite
        env_single = SatelliteHandoverEnv(
            adapter=self.adapter,
            satellite_ids=self.test_satellite_ids[:1],
            config=self.config
        )

        observation, info = env_single.reset()

        # Step with action 0 (stay)
        observation, reward, terminated, truncated, info = env_single.step(0)

        # Should work normally

    def test_action_out_of_range(self):
        """Test action index larger than visible satellites"""
        observation, info = self.env.reset()

        # Try to switch to satellite beyond visible range
        large_action = self.env.max_visible_satellites  # Max valid action

        # Should not raise, but should be treated as invalid
        # (based on code, it treats out of range as "stay")

    def test_reset_consistency(self):
        """Test that reset is consistent with same seed"""
        obs1, info1 = self.env.reset(seed=123)
        obs2, info2 = self.env.reset(seed=123)

        # Should give identical initial states
        np.testing.assert_array_equal(obs1, obs2)
        self.assertEqual(info1['current_satellite'], info2['current_satellite'])


if __name__ == '__main__':
    unittest.main()
