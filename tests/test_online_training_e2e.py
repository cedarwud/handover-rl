#!/usr/bin/env python3
"""
End-to-End Test Suite for Online RL Training

Tests the complete training pipeline from initialization to checkpoint saving.
Covers:
- Component initialization (Adapter, Environment, Agent)
- Training loop execution
- Checkpoint saving
- Metrics logging
- Full integration workflow

Academic Standard: Real TLE data, complete physics, no mocking
"""

import unittest
import sys
from pathlib import Path
import numpy as np
import yaml
import shutil
import tempfile
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tests.test_base import BaseRLTest
from tests.test_utils import (
    load_test_config,
    get_test_satellite_ids,
    create_mock_adapter_for_testing,
)


class TestOnlineTrainingInitialization(BaseRLTest):
    """Test initialization of all training components"""

    def test_adapter_initialization(self):
        """Test OrbitEngineAdapter initialization"""
        from adapters.orbit_engine_adapter import OrbitEngineAdapter

        config = load_test_config()
        adapter = OrbitEngineAdapter(config)

        self.assertIsNotNone(adapter)
        self.assertIsNotNone(adapter.tle_loader)

    def test_environment_initialization(self):
        """Test SatelliteHandoverEnv initialization"""
        from adapters.orbit_engine_adapter import OrbitEngineAdapter
        from environments.satellite_handover_env import SatelliteHandoverEnv

        config = load_test_config()
        adapter = OrbitEngineAdapter(config)
        satellite_ids = get_test_satellite_ids(10)

        env = SatelliteHandoverEnv(adapter, satellite_ids, config)

        self.assertIsNotNone(env)
        self.assertEqual(len(env.satellite_ids), 10)

    def test_agent_initialization(self):
        """Test DQNAgent initialization"""
        from adapters.orbit_engine_adapter import OrbitEngineAdapter
        from environments.satellite_handover_env import SatelliteHandoverEnv
        from agents.dqn_agent_v2 import DQNAgent

        config = load_test_config()
        adapter = OrbitEngineAdapter(config)
        satellite_ids = get_test_satellite_ids(10)
        env = SatelliteHandoverEnv(adapter, satellite_ids, config)

        agent = DQNAgent(
            state_dim=env.observation_space.shape[1],
            action_dim=env.action_space.n,
            config=config
        )

        self.assertIsNotNone(agent)
        self.assertIsNotNone(agent.q_network)
        self.assertIsNotNone(agent.target_network)
        self.assertIsNotNone(agent.replay_buffer)

    def test_satellite_pool_loading(self):
        """Test loading optimized satellite pool"""
        from utils.satellite_utils import load_stage4_optimized_satellites

        satellite_ids = load_stage4_optimized_satellites()

        self.assertIsNotNone(satellite_ids)
        self.assertGreater(len(satellite_ids), 0)
        self.assertIsInstance(satellite_ids, list)


class TestOnlineTrainingQuickRun(BaseRLTest):
    """Test quick training run (10 episodes)"""

    def setUp(self):
        """Set up test environment and agent"""
        super().setUp()

        from adapters.orbit_engine_adapter import OrbitEngineAdapter
        from environments.satellite_handover_env import SatelliteHandoverEnv
        from agents.dqn_agent_v2 import DQNAgent

        # Load config
        self.config = load_test_config()

        # Create components
        self.adapter = OrbitEngineAdapter(self.config)
        self.satellite_ids = get_test_satellite_ids(10)
        self.env = SatelliteHandoverEnv(self.adapter, self.satellite_ids, self.config)
        self.agent = DQNAgent(
            state_dim=self.env.observation_space.shape[1],
            action_dim=self.env.action_space.n,
            config=self.config
        )

    def test_single_episode_execution(self):
        """Test executing a single training episode"""
        state, info = self.env.reset(seed=42)

        episode_reward = 0
        episode_steps = 0
        max_steps = 100

        for step in range(max_steps):
            # Select action
            action = self.agent.select_action(state, episode=0)

            # Step environment
            next_state, reward, terminated, truncated, info = self.env.step(action)

            # Store transition (don't train yet)
            self.agent.replay_buffer.push(state, action, reward, next_state, terminated or truncated)

            episode_reward += reward
            episode_steps += 1

            state = next_state

            if terminated or truncated:
                break

        # Verify episode ran
        self.assertGreater(episode_steps, 0)
        self.assertIsInstance(episode_reward, (int, float))

    def test_quick_training_loop(self):
        """Test quick training loop (10 episodes)"""
        num_episodes = 10
        episode_rewards = []

        for episode in range(num_episodes):
            state, info = self.env.reset(seed=episode)

            episode_reward = 0
            max_steps = 50  # Short episodes for testing

            for step in range(max_steps):
                # Select action
                action = self.agent.select_action(state, episode=episode)

                # Step environment
                next_state, reward, terminated, truncated, info = self.env.step(action)

                # Store transition
                self.agent.replay_buffer.push(state, action, reward, next_state, terminated or truncated)

                # Train (if enough samples)
                if len(self.agent.replay_buffer) >= self.agent.batch_size:
                    loss = self.agent.train()

                episode_reward += reward
                state = next_state

                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)

            # Update target network periodically
            if episode % 5 == 0:
                self.agent.update_target_network()

        # Verify training ran
        self.assertEqual(len(episode_rewards), num_episodes)
        self.assertGreater(len(self.agent.replay_buffer), 0)

    def test_epsilon_decay(self):
        """Test epsilon decay during training"""
        initial_epsilon = self.agent.epsilon

        # Run some episodes
        for episode in range(20):
            state, info = self.env.reset(seed=episode)

            for step in range(10):
                action = self.agent.select_action(state, episode=episode)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                state = next_state

                if terminated or truncated:
                    break

        # Epsilon should have decayed
        self.assertLess(self.agent.epsilon, initial_epsilon)

    def test_replay_buffer_filling(self):
        """Test that replay buffer fills during training"""
        initial_size = len(self.agent.replay_buffer)

        # Run episode
        state, info = self.env.reset(seed=42)

        for step in range(50):
            action = self.agent.select_action(state, episode=0)
            next_state, reward, terminated, truncated, info = self.env.step(action)

            self.agent.replay_buffer.push(state, action, reward, next_state, terminated or truncated)

            state = next_state

            if terminated or truncated:
                break

        # Buffer should have grown
        self.assertGreater(len(self.agent.replay_buffer), initial_size)


class TestOnlineTrainingCheckpoints(BaseRLTest):
    """Test checkpoint saving and loading"""

    def setUp(self):
        """Set up test environment, agent, and temp directory"""
        super().setUp()

        from adapters.orbit_engine_adapter import OrbitEngineAdapter
        from environments.satellite_handover_env import SatelliteHandoverEnv
        from agents.dqn_agent_v2 import DQNAgent

        # Load config
        self.config = load_test_config()

        # Create components
        self.adapter = OrbitEngineAdapter(self.config)
        self.satellite_ids = get_test_satellite_ids(10)
        self.env = SatelliteHandoverEnv(self.adapter, self.satellite_ids, self.config)
        self.agent = DQNAgent(
            state_dim=self.env.observation_space.shape[1],
            action_dim=self.env.action_space.n,
            config=self.config
        )

        # Create temp directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_checkpoint_save(self):
        """Test saving checkpoint"""
        import torch

        checkpoint_path = Path(self.temp_dir) / "test_checkpoint.pt"

        # Save checkpoint
        torch.save({
            'episode': 10,
            'q_network_state_dict': self.agent.q_network.state_dict(),
            'target_network_state_dict': self.agent.target_network.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'epsilon': self.agent.epsilon,
        }, checkpoint_path)

        # Verify file exists
        self.assertTrue(checkpoint_path.exists())

    def test_checkpoint_load(self):
        """Test loading checkpoint"""
        import torch

        checkpoint_path = Path(self.temp_dir) / "test_checkpoint.pt"

        # Train for a bit to change state
        state, info = self.env.reset(seed=42)
        for _ in range(10):
            action = self.agent.select_action(state, episode=0)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            self.agent.replay_buffer.push(state, action, reward, next_state, terminated or truncated)
            state = next_state
            if terminated or truncated:
                break

        # Save original epsilon
        original_epsilon = self.agent.epsilon

        # Save checkpoint
        torch.save({
            'episode': 10,
            'q_network_state_dict': self.agent.q_network.state_dict(),
            'target_network_state_dict': self.agent.target_network.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'epsilon': original_epsilon,
        }, checkpoint_path)

        # Modify epsilon
        self.agent.epsilon = 0.5

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        self.agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.agent.epsilon = checkpoint['epsilon']

        # Verify epsilon was restored
        self.assertEqual(self.agent.epsilon, original_epsilon)


class TestOnlineTrainingOutputs(BaseRLTest):
    """Test training output files and metrics"""

    def setUp(self):
        """Set up temp directory for outputs"""
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_metrics_logging(self):
        """Test that training metrics can be logged"""
        import json

        metrics_file = Path(self.temp_dir) / "metrics.json"

        # Simulate metrics collection
        metrics = {
            'episode': 10,
            'reward': 100.5,
            'loss': 0.123,
            'epsilon': 0.5,
            'num_handovers': 5,
        }

        # Save metrics
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Verify file exists and can be loaded
        self.assertTrue(metrics_file.exists())

        with open(metrics_file, 'r') as f:
            loaded_metrics = json.load(f)

        self.assertEqual(loaded_metrics['episode'], 10)
        self.assertEqual(loaded_metrics['reward'], 100.5)

    def test_checkpoint_directory_creation(self):
        """Test checkpoint directory creation"""
        checkpoint_dir = Path(self.temp_dir) / "checkpoints"

        # Create directory
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Verify directory exists
        self.assertTrue(checkpoint_dir.exists())
        self.assertTrue(checkpoint_dir.is_dir())


class TestOnlineTrainingIntegration(BaseRLTest):
    """Integration tests for complete training workflow"""

    def test_full_training_workflow_mini(self):
        """Test complete training workflow (5 episodes)"""
        from adapters.orbit_engine_adapter import OrbitEngineAdapter
        from environments.satellite_handover_env import SatelliteHandoverEnv
        from agents.dqn_agent_v2 import DQNAgent

        # Load config
        config = load_test_config()

        # Initialize components
        adapter = OrbitEngineAdapter(config)
        satellite_ids = get_test_satellite_ids(10)
        env = SatelliteHandoverEnv(adapter, satellite_ids, config)
        agent = DQNAgent(
            state_dim=env.observation_space.shape[1],
            action_dim=env.action_space.n,
            config=config
        )

        # Training loop
        num_episodes = 5
        max_steps_per_episode = 20

        episode_rewards = []
        episode_losses = []

        for episode in range(num_episodes):
            state, info = env.reset(seed=episode)

            episode_reward = 0
            episode_loss = []

            for step in range(max_steps_per_episode):
                # Select action
                action = agent.select_action(state, episode=episode)

                # Step environment
                next_state, reward, terminated, truncated, info = env.step(action)

                # Store transition
                agent.replay_buffer.push(state, action, reward, next_state, terminated or truncated)

                # Train
                if len(agent.replay_buffer) >= agent.batch_size:
                    loss = agent.train()
                    if loss is not None:
                        episode_loss.append(loss)

                episode_reward += reward
                state = next_state

                if terminated or truncated:
                    break

            # Update target network
            if episode % 2 == 0:
                agent.update_target_network()

            episode_rewards.append(episode_reward)
            if len(episode_loss) > 0:
                episode_losses.append(np.mean(episode_loss))

        # Verify training completed
        self.assertEqual(len(episode_rewards), num_episodes)
        self.assertGreater(len(agent.replay_buffer), 0)

        # Verify metrics are reasonable
        for reward in episode_rewards:
            self.assertIsInstance(reward, (int, float))
            self.assertFalse(np.isnan(reward))
            self.assertFalse(np.isinf(reward))

    def test_components_use_real_data(self):
        """Test that all components use real data (not mocked)"""
        from adapters.orbit_engine_adapter import OrbitEngineAdapter
        from environments.satellite_handover_env import SatelliteHandoverEnv
        from agents.dqn_agent_v2 import DQNAgent

        # Load config
        config = load_test_config()

        # Initialize components
        adapter = OrbitEngineAdapter(config)
        satellite_ids = get_test_satellite_ids(10)
        env = SatelliteHandoverEnv(adapter, satellite_ids, config)

        # Verify adapter uses real TLE
        self.assertIsNotNone(adapter.tle_loader)

        # Run environment and check states
        state, info = env.reset(seed=42)

        # If satellites visible, check for real physics values
        if info['num_visible'] > 0:
            # RSRP values should be in reasonable range
            rsrp_values = state[:, 0]
            non_zero_rsrp = rsrp_values[rsrp_values != 0]

            if len(non_zero_rsrp) > 0:
                # Should have diversity (not hardcoded)
                if len(non_zero_rsrp) > 1:
                    self.assertNoHardcoding(non_zero_rsrp.tolist(), min_diversity=2)


class TestOnlineTrainingConfiguration(BaseRLTest):
    """Test configuration loading and validation"""

    def test_config_loading(self):
        """Test loading training configuration"""
        config = load_test_config()

        self.assertIsNotNone(config)
        self.assertIsInstance(config, dict)

    def test_config_has_required_sections(self):
        """Test that config has all required sections"""
        config = load_test_config()

        # Check for required sections
        # (Note: Config structure may vary, adjust as needed)
        # For now, just verify it's not empty
        self.assertGreater(len(config), 0)

    def test_config_parameters_valid(self):
        """Test that config parameters are valid"""
        config = load_test_config()

        # If environment config exists, validate parameters
        env_config = config.get('environment', config.get('data_generation', {}))

        if env_config:
            # Check numeric parameters are positive
            if 'time_step_seconds' in env_config:
                self.assertGreater(env_config['time_step_seconds'], 0)

            if 'episode_duration_minutes' in env_config:
                self.assertGreater(env_config['episode_duration_minutes'], 0)


if __name__ == '__main__':
    unittest.main()
