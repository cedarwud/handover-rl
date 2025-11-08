#!/usr/bin/env python3
"""
Integration Tests - End-to-End Pipeline Testing

Tests for the complete Handover-RL V2.0 framework integration.

Test Coverage:
    - Phase 1 + Phase 2 + Phase 3a integration (DQN training with dummy env)
    - Phase 1 + Phase 3b integration (data generation)
    - Phase 2 + Phase 3c integration (training with real episodes)
    - Full pipeline: Data generation → Training → Evaluation

Run tests:
    cd /home/sat/satellite/handover-rl
    python tests/test_integration.py
"""

import unittest
import numpy as np
import tempfile
import shutil
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

# Import modules to test
try:
    from rl_core import BaseRLAgent, DummyAgent, BaseHandoverEnvironment, UniversalRLTrainer
    from environments import HandoverEnvironment
    from data_generation import EpisodeBuilder, load_episodes
    RL_CORE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import RL core modules: {e}")
    RL_CORE_AVAILABLE = False

try:
    from agents import DQNAgent
    DQN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import DQN agent (PyTorch required): {e}")
    DQN_AVAILABLE = False


@unittest.skipIf(not RL_CORE_AVAILABLE, "RL core modules not available")
class TestPhase1And2Integration(unittest.TestCase):
    """Test integration of Phase 1 (Adapters) and Phase 2 (RL Core)."""

    def test_dummy_agent_with_base_env(self):
        """Test DummyAgent with BaseHandoverEnvironment."""
        config = {
            'environment': {
                'state_dim': 12,
                'action_dim': 2,
                'max_steps_per_episode': 50
            }
        }

        agent = DummyAgent(state_dim=12, action_dim=2, config={})
        env = BaseHandoverEnvironment(config)

        # Run one episode
        state, info = env.reset()
        total_reward = 0

        for step in range(10):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        self.assertGreater(step, 0)
        print(f"✅ DummyAgent + BaseEnv: {step+1} steps, reward={total_reward:.2f}")


@unittest.skipIf(not RL_CORE_AVAILABLE, "RL core modules not available")
class TestPhase2And3Integration(unittest.TestCase):
    """Test integration of Phase 2 (RL Core) and Phase 3 (Environments)."""

    def setUp(self):
        """Create test episode."""
        T = 100
        self.test_episode = {
            'states': np.random.randn(T, 12).astype(np.float32),
            'actions': np.random.randint(0, 2, T),
            'rewards': np.random.randn(T).astype(np.float32),
            'next_states': np.random.randn(T, 12).astype(np.float32),
            'dones': np.zeros(T, dtype=np.float32),
            'timestamps': np.array([1704067200 + i * 5 for i in range(T)], dtype=np.float64),
            'metadata': {'episode_id': 0}
        }
        self.test_episode['dones'][-1] = 1.0
        self.test_episode['states'][:, 0] = np.random.uniform(-110, -60, T)  # Valid RSRP

    def test_handover_env_with_episode(self):
        """Test HandoverEnvironment with loaded episode."""
        config = {
            'environment': {
                'state_dim': 12,
                'action_dim': 2,
                'max_steps_per_episode': 1500
            }
        }

        env = HandoverEnvironment(config, episodes=[self.test_episode])

        # Reset and check
        state, info = env.reset()
        self.assertEqual(state.shape, (12,))
        self.assertIn('episode_idx', info)

        # Take steps
        for step in range(10):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)

            self.assertEqual(next_state.shape, (12,))
            self.assertIsInstance(reward, float)

            if terminated or truncated:
                break

        print(f"✅ HandoverEnv with episode: {step+1} steps completed")

    def test_dummy_agent_with_handover_env(self):
        """Test DummyAgent with HandoverEnvironment."""
        config = {
            'environment': {
                'state_dim': 12,
                'action_dim': 2,
                'max_steps_per_episode': 1500
            }
        }

        agent = DummyAgent(state_dim=12, action_dim=2, config={})
        env = HandoverEnvironment(config, episodes=[self.test_episode])

        state, info = env.reset()
        total_reward = 0

        for step in range(20):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        self.assertGreater(step, 0)
        print(f"✅ DummyAgent + HandoverEnv: {step+1} steps, reward={total_reward:.2f}")


@unittest.skipIf(not DQN_AVAILABLE, "DQN agent not available (PyTorch required)")
class TestDQNIntegration(unittest.TestCase):
    """Test DQN agent integration with environments."""

    def test_dqn_with_base_env_training(self):
        """Test DQN training with BaseHandoverEnvironment."""
        agent_config = {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.1,
            'epsilon_decay': 0.99,
            'batch_size': 32,
            'buffer_size': 500,
            'target_update_frequency': 50,
            'use_double_dqn': True,
            'hidden_dims': [64, 64]
        }

        config = {
            'environment': {
                'state_dim': 12,
                'action_dim': 2,
                'max_steps_per_episode': 50
            },
            'training': {
                'epochs': 2,
                'episodes_per_epoch': 5,
                'checkpoint': {'save_frequency': 100},
                'logging': {'tensorboard': False, 'wandb': False}
            }
        }

        agent = DQNAgent(state_dim=12, action_dim=2, config=agent_config)
        env = BaseHandoverEnvironment(config)

        # Run a few episodes
        for episode in range(3):
            state, info = env.reset()
            total_reward = 0

            for step in range(50):
                action = agent.select_action(state, eval_mode=False)
                next_state, reward, terminated, truncated, info = env.step(action)

                # Store in replay buffer
                agent.on_step(state, action, reward, next_state, terminated or truncated)

                # Update agent
                if step > 32:  # Wait for enough samples
                    loss_info = agent.update()

                total_reward += reward
                state = next_state

                if terminated or truncated:
                    break

            print(f"   Episode {episode+1}: {step+1} steps, reward={total_reward:.2f}")

        buffer_size = len(agent.replay_buffer)
        self.assertGreater(buffer_size, 0)
        print(f"✅ DQN training test: buffer size={buffer_size}")


@unittest.skipIf(not RL_CORE_AVAILABLE, "RL core modules not available")
class TestUniversalTrainerIntegration(unittest.TestCase):
    """Test UniversalRLTrainer integration."""

    def test_trainer_with_dummy_agent(self):
        """Test UniversalRLTrainer with DummyAgent."""
        config = {
            'environment': {
                'state_dim': 12,
                'action_dim': 2,
                'max_steps_per_episode': 50
            },
            'training': {
                'epochs': 2,
                'episodes_per_epoch': 3,
                'checkpoint': {'save_frequency': 100},
                'validation': {'frequency': 10, 'episodes': 2},
                'logging': {'log_frequency': 1, 'tensorboard': False, 'wandb': False}
            }
        }

        agent = DummyAgent(state_dim=12, action_dim=2, config={})
        train_env = BaseHandoverEnvironment(config)
        val_env = BaseHandoverEnvironment(config)

        trainer = UniversalRLTrainer(agent, train_env, val_env, config)

        # Train
        history = trainer.train()

        self.assertGreater(len(history), 0)
        self.assertEqual(len(history), config['training']['epochs'] * config['training']['episodes_per_epoch'])

        print(f"✅ UniversalTrainer: {len(history)} episodes trained")


@unittest.skipIf(not RL_CORE_AVAILABLE, "RL core modules not available")
class TestEpisodeBuilderIntegration(unittest.TestCase):
    """Test EpisodeBuilder integration."""

    def test_episode_building_and_validation(self):
        """Test building and validating episodes."""
        builder = EpisodeBuilder()

        # Create episode
        T = 100
        states = np.random.randn(T, 12).astype(np.float32)
        states[:, 0] = np.random.uniform(-110, -60, T)  # Valid RSRP
        actions = np.random.randint(0, 2, T)
        rewards = np.random.randn(T).astype(np.float32)
        timestamps = np.array([1704067200 + i * 5 for i in range(T)], dtype=np.float64)

        episode = builder.build_episode(states, actions, rewards, timestamps)

        # Validate
        is_valid, info = builder.validate_episode(episode)

        self.assertTrue(is_valid)
        self.assertGreater(info['valid_ratio'], 0.5)

        # Get statistics
        stats = builder.get_statistics(episode)

        self.assertIn('total_reward', stats)
        self.assertIn('handover_rate', stats)

        print(f"✅ Episode building: valid_ratio={info['valid_ratio']:.2%}, "
              f"handover_rate={stats['handover_rate']:.2%}")


# Main test runner
if __name__ == "__main__":
    print("=" * 70)
    print("Integration Tests - Handover-RL V2.0")
    print("=" * 70)
    print()

    # Run tests
    unittest.main(verbosity=2)
