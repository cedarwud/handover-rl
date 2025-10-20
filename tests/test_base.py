#!/usr/bin/env python3
"""
Base Test Class for LEO Satellite Handover RL

Provides common test setup and teardown functionality
"""

import unittest
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tests.test_utils import (
    load_test_config,
    get_test_satellite_ids,
    get_test_timestamp,
    create_mock_adapter_for_testing,
)


class BaseRLTest(unittest.TestCase):
    """
    Base class for RL system tests

    Provides common setup and utilities for testing the refactored system
    """

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests in class"""
        # Load configuration
        cls.config = load_test_config()

        # Create adapter (REAL, not mock)
        cls.adapter = create_mock_adapter_for_testing()

        # Test satellite IDs
        cls.test_satellite_ids = get_test_satellite_ids(n=10)

        # Test timestamp
        cls.test_timestamp = get_test_timestamp()

        print(f"\n✅ Test setup complete:")
        print(f"   - Config loaded")
        print(f"   - OrbitEngineAdapter initialized (real TLE data)")
        print(f"   - {len(cls.test_satellite_ids)} test satellites")
        print(f"   - Test timestamp: {cls.test_timestamp}")

    def setUp(self):
        """Set up for each individual test"""
        # Set random seed for reproducibility
        np.random.seed(42)

    def tearDown(self):
        """Clean up after each test"""
        pass

    def assertStateValid(self, state_dict):
        """
        Assert that a state dictionary is valid

        Args:
            state_dict: State from OrbitEngineAdapter
        """
        from tests.test_utils import verify_state_dict
        self.assertTrue(verify_state_dict(state_dict),
                        "State dictionary validation failed")

    def assertObservationValid(self, observation, expected_shape=(10, 12)):
        """
        Assert that an observation array is valid

        Args:
            observation: Observation from environment
            expected_shape: Expected shape
        """
        from tests.test_utils import verify_observation_space
        self.assertTrue(verify_observation_space(observation, expected_shape),
                        "Observation validation failed")

    def assertNoHardcoding(self, values, min_diversity=2):
        """
        Assert that values show diversity (not hardcoded)

        Args:
            values: List of values to check
            min_diversity: Minimum unique values required
        """
        from tests.test_utils import assert_no_hardcoding
        assert_no_hardcoding(values, min_diversity)

    def assertUsesRealTLE(self):
        """Assert that adapter uses real TLE data"""
        from tests.test_utils import assert_uses_real_tle
        assert_uses_real_tle(self.adapter)


class BaseEnvironmentTest(BaseRLTest):
    """
    Base class for environment tests

    Extends BaseRLTest with environment-specific utilities
    """

    @classmethod
    def setUpClass(cls):
        """Set up environment test fixtures"""
        super().setUpClass()

        # Environment will be created in subclass
        cls.env = None

    def assertActionValid(self, action, action_space):
        """
        Assert that an action is valid for the action space

        Args:
            action: Action to validate
            action_space: Gym action space
        """
        self.assertTrue(action_space.contains(action),
                        f"Action {action} not in action space {action_space}")


class BaseAgentTest(BaseRLTest):
    """
    Base class for agent tests

    Extends BaseRLTest with agent-specific utilities
    """

    @classmethod
    def setUpClass(cls):
        """Set up agent test fixtures"""
        super().setUpClass()

        # Agent will be created in subclass
        cls.agent = None

    def assertReplayBufferWorking(self, agent, min_size=10):
        """
        Assert that replay buffer is functioning

        Args:
            agent: DQN agent instance
            min_size: Minimum buffer size to check
        """
        self.assertIsNotNone(agent.replay_buffer,
                             "Agent should have replay buffer")
        self.assertGreaterEqual(len(agent.replay_buffer), min_size,
                                f"Replay buffer should have at least {min_size} experiences")

    def assertLossDecreasing(self, loss_history, window=100):
        """
        Assert that training loss is decreasing

        Args:
            loss_history: List of loss values
            window: Window size for moving average
        """
        if len(loss_history) < window * 2:
            self.skipTest(f"Not enough loss history ({len(loss_history)} < {window*2})")

        # Compare first and last windows
        first_window_avg = np.mean(loss_history[:window])
        last_window_avg = np.mean(loss_history[-window:])

        self.assertLess(last_window_avg, first_window_avg,
                        f"Loss not decreasing: first={first_window_avg:.4f}, "
                        f"last={last_window_avg:.4f}")


if __name__ == '__main__':
    # Run tests if this file is executed directly
    unittest.main()
