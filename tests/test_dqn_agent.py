#!/usr/bin/env python3
"""
Unit Tests for DQN Agent

Tests for DQN network, replay buffer, and DQN agent implementation.

Test Coverage:
    - DQN Network: Forward pass, action selection, Q-value computation
    - Dueling DQN Network: Architecture verification
    - Replay Buffer: Push, sample, statistics
    - Prioritized Replay Buffer: Priority-based sampling
    - DQN Agent: Initialization, action selection, update, save/load

Run tests:
    cd /home/sat/satellite/handover-rl
    python tests/test_dqn_agent.py
"""

import unittest
import numpy as np
import torch
import sys
import tempfile
import os
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

# Import modules to test
try:
    from agents.dqn_network import DQNNetwork, DuelingDQNNetwork
    from agents.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
    from agents.dqn_agent import DQNAgent
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import DQN modules: {e}")
    MODULES_AVAILABLE = False


@unittest.skipIf(not MODULES_AVAILABLE, "DQN modules not available")
class TestDQNNetwork(unittest.TestCase):
    """Test DQN Network Architecture."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_dim = 12
        self.action_dim = 2
        self.hidden_dims = [128, 128]

    def test_network_initialization(self):
        """Test network initialization."""
        network = DQNNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims
        )

        # Check attributes
        self.assertEqual(network.state_dim, self.state_dim)
        self.assertEqual(network.action_dim, self.action_dim)
        self.assertEqual(network.hidden_dims, self.hidden_dims)

        # Check parameter count
        param_count = sum(p.numel() for p in network.parameters())
        self.assertGreater(param_count, 0)

        print(f"✅ DQN Network initialized: {param_count} parameters")

    def test_forward_pass_single(self):
        """Test forward pass with single state."""
        network = DQNNetwork(self.state_dim, self.action_dim, self.hidden_dims)

        # Single state
        state = torch.randn(self.state_dim)
        q_values = network(state)

        # Check output shape
        self.assertEqual(q_values.shape, (self.action_dim,))

        print(f"✅ Forward pass (single): input {state.shape} -> output {q_values.shape}")

    def test_forward_pass_batch(self):
        """Test forward pass with batch of states."""
        network = DQNNetwork(self.state_dim, self.action_dim, self.hidden_dims)

        # Batch of states
        batch_size = 32
        states = torch.randn(batch_size, self.state_dim)
        q_values = network(states)

        # Check output shape
        self.assertEqual(q_values.shape, (batch_size, self.action_dim))

        print(f"✅ Forward pass (batch): input {states.shape} -> output {q_values.shape}")

    def test_action_selection(self):
        """Test greedy action selection."""
        network = DQNNetwork(self.state_dim, self.action_dim, self.hidden_dims)

        # Test with numpy array
        state_np = np.random.randn(self.state_dim).astype(np.float32)
        action = network.get_action(state_np)

        # Check action is valid
        self.assertIn(action, [0, 1])

        # Test with tensor
        state_tensor = torch.randn(self.state_dim)
        action = network.get_action(state_tensor)
        self.assertIn(action, [0, 1])

        print(f"✅ Action selection: action={action}")

    def test_q_value_retrieval(self):
        """Test Q-value retrieval for state-action pair."""
        network = DQNNetwork(self.state_dim, self.action_dim, self.hidden_dims)

        state = np.random.randn(self.state_dim).astype(np.float32)

        # Get Q-values for both actions
        q_value_0 = network.get_q_value(state, 0)
        q_value_1 = network.get_q_value(state, 1)

        self.assertIsInstance(q_value_0, float)
        self.assertIsInstance(q_value_1, float)

        print(f"✅ Q-value retrieval: Q(s,0)={q_value_0:.3f}, Q(s,1)={q_value_1:.3f}")


@unittest.skipIf(not MODULES_AVAILABLE, "DQN modules not available")
class TestDuelingDQN(unittest.TestCase):
    """Test Dueling DQN Network."""

    def test_dueling_architecture(self):
        """Test dueling architecture forward pass."""
        network = DuelingDQNNetwork(state_dim=12, action_dim=2, hidden_dims=[128, 128])

        # Forward pass
        state = torch.randn(12)
        q_values = network(state)

        # Check output
        self.assertEqual(q_values.shape, (2,))

        # Batch forward pass
        states = torch.randn(32, 12)
        q_values_batch = network(states)
        self.assertEqual(q_values_batch.shape, (32, 2))

        print(f"✅ Dueling DQN verified: Q-values shape {q_values.shape}")


@unittest.skipIf(not MODULES_AVAILABLE, "DQN modules not available")
class TestReplayBuffer(unittest.TestCase):
    """Test Replay Buffer."""

    def setUp(self):
        """Set up test fixtures."""
        self.capacity = 1000
        self.state_dim = 12

    def test_buffer_initialization(self):
        """Test buffer initialization."""
        buffer = ReplayBuffer(capacity=self.capacity)

        self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer.capacity, self.capacity)

        print(f"✅ Replay buffer initialized: capacity={self.capacity}")

    def test_push_transitions(self):
        """Test adding transitions."""
        buffer = ReplayBuffer(capacity=self.capacity)

        # Add transitions
        for i in range(10):
            state = np.random.randn(self.state_dim).astype(np.float32)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_state = np.random.randn(self.state_dim).astype(np.float32)
            done = False

            buffer.push(state, action, reward, next_state, done)

        self.assertEqual(len(buffer), 10)

        print(f"✅ Added 10 transitions: buffer size={len(buffer)}")

    def test_sampling(self):
        """Test batch sampling."""
        buffer = ReplayBuffer(capacity=self.capacity, seed=42)

        # Add transitions
        for i in range(100):
            state = np.random.randn(self.state_dim).astype(np.float32)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_state = np.random.randn(self.state_dim).astype(np.float32)
            done = (i % 20 == 19)

            buffer.push(state, action, reward, next_state, done)

        # Sample batch
        batch_size = 32
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        # Check shapes
        self.assertEqual(states.shape, (batch_size, self.state_dim))
        self.assertEqual(actions.shape, (batch_size,))
        self.assertEqual(rewards.shape, (batch_size,))
        self.assertEqual(next_states.shape, (batch_size, self.state_dim))
        self.assertEqual(dones.shape, (batch_size,))

        print(f"✅ Sampled batch: {batch_size} transitions")

    def test_buffer_statistics(self):
        """Test buffer statistics."""
        buffer = ReplayBuffer(capacity=self.capacity)

        # Add transitions
        for i in range(50):
            state = np.random.randn(self.state_dim).astype(np.float32)
            action = i % 2  # Alternate actions
            reward = np.random.randn()
            next_state = np.random.randn(self.state_dim).astype(np.float32)
            done = False

            buffer.push(state, action, reward, next_state, done)

        # Get statistics
        stats = buffer.get_statistics()

        self.assertEqual(stats['size'], 50)
        self.assertEqual(stats['capacity'], self.capacity)
        self.assertGreater(stats['utilization'], 0)
        self.assertIn('avg_reward', stats)
        self.assertEqual(len(stats['action_distribution']), 2)

        print(f"✅ Buffer statistics: {stats['size']}/{stats['capacity']} "
              f"({stats['utilization']:.1%} utilization)")


@unittest.skipIf(not MODULES_AVAILABLE, "DQN modules not available")
class TestPrioritizedReplayBuffer(unittest.TestCase):
    """Test Prioritized Replay Buffer."""

    def test_prioritized_sampling(self):
        """Test priority-based sampling."""
        buffer = PrioritizedReplayBuffer(
            capacity=1000,
            alpha=0.6,
            beta=0.4,
            seed=42
        )

        # Add transitions with priorities
        for i in range(100):
            state = np.random.randn(12).astype(np.float32)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_state = np.random.randn(12).astype(np.float32)
            done = False
            priority = np.random.rand()

            buffer.push(state, action, reward, next_state, done, priority)

        # Sample batch
        batch_size = 32
        result = buffer.sample(batch_size)
        states, actions, rewards, next_states, dones, indices, weights = result

        # Check shapes
        self.assertEqual(len(states), batch_size)
        self.assertEqual(len(indices), batch_size)
        self.assertEqual(len(weights), batch_size)

        # Check weights are normalized
        self.assertAlmostEqual(weights.max(), 1.0, places=5)

        print(f"✅ Prioritized sampling: batch_size={batch_size}, "
              f"weights range=[{weights.min():.3f}, {weights.max():.3f}]")


@unittest.skipIf(not MODULES_AVAILABLE, "DQN modules not available")
class TestDQNAgent(unittest.TestCase):
    """Test DQN Agent."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'batch_size': 32,
            'buffer_size': 1000,
            'target_update_frequency': 100,
            'use_double_dqn': True,
            'use_dueling': False,
            'use_prioritized_replay': False,
            'hidden_dims': [64, 64]
        }

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = DQNAgent(state_dim=12, action_dim=2, config=self.config)

        # Check attributes
        self.assertEqual(agent.state_dim, 12)
        self.assertEqual(agent.action_dim, 2)
        self.assertEqual(agent.epsilon, self.config['epsilon_start'])
        self.assertEqual(len(agent.replay_buffer), 0)

        print(f"✅ DQN Agent initialized: epsilon={agent.epsilon}")

    def test_action_selection_exploration(self):
        """Test action selection in exploration mode."""
        agent = DQNAgent(state_dim=12, action_dim=2, config=self.config)

        state = np.random.randn(12).astype(np.float32)

        # Exploration mode
        action = agent.select_action(state, eval_mode=False)
        self.assertIn(action, [0, 1])

        print(f"✅ Exploration action: {action}")

    def test_action_selection_evaluation(self):
        """Test action selection in evaluation mode."""
        agent = DQNAgent(state_dim=12, action_dim=2, config=self.config)

        state = np.random.randn(12).astype(np.float32)

        # Evaluation mode (greedy)
        action = agent.select_action(state, eval_mode=True)
        self.assertIn(action, [0, 1])

        print(f"✅ Evaluation action: {action}")

    def test_replay_buffer_storage(self):
        """Test storing transitions in replay buffer."""
        agent = DQNAgent(state_dim=12, action_dim=2, config=self.config)

        # Add transitions
        for i in range(50):
            state = np.random.randn(12).astype(np.float32)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_state = np.random.randn(12).astype(np.float32)
            done = False

            agent.on_step(state, action, reward, next_state, done)

        self.assertEqual(len(agent.replay_buffer), 50)

        print(f"✅ Replay buffer storage: {len(agent.replay_buffer)} transitions")

    def test_update(self):
        """Test agent update."""
        agent = DQNAgent(state_dim=12, action_dim=2, config=self.config)

        # Add enough transitions for training
        for i in range(100):
            state = np.random.randn(12).astype(np.float32)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_state = np.random.randn(12).astype(np.float32)
            done = False

            agent.on_step(state, action, reward, next_state, done)

        # Update agent
        loss_info = agent.update()

        # Check loss info
        self.assertIn('total_loss', loss_info)
        self.assertIn('epsilon', loss_info)
        self.assertIsInstance(loss_info['total_loss'], float)

        print(f"✅ Agent update: loss={loss_info['total_loss']:.4f}")

    def test_save_load(self):
        """Test save and load checkpoint."""
        agent = DQNAgent(state_dim=12, action_dim=2, config=self.config)

        # Set some state
        agent.epsilon = 0.5
        agent.total_steps = 1000

        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name

        try:
            agent.save(checkpoint_path)

            # Create new agent and load
            new_agent = DQNAgent(state_dim=12, action_dim=2, config=self.config)
            new_agent.load(checkpoint_path)

            # Check loaded state
            self.assertAlmostEqual(new_agent.epsilon, 0.5, places=5)
            self.assertEqual(new_agent.total_steps, 1000)

            print(f"✅ Save/Load: epsilon={new_agent.epsilon}, steps={new_agent.total_steps}")

        finally:
            # Clean up
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

    def test_training_state(self):
        """Test getting training state."""
        agent = DQNAgent(state_dim=12, action_dim=2, config=self.config)

        state = agent.get_training_state()

        # Check required fields
        self.assertIn('total_steps', state)
        self.assertIn('total_episodes', state)
        self.assertIn('epsilon', state)
        self.assertIn('buffer_size', state)

        print(f"✅ Training state: {len(state)} fields")


# Main test runner
if __name__ == "__main__":
    print("=" * 70)
    print("DQN Agent Unit Tests")
    print("=" * 70)

    # Run tests
    unittest.main(verbosity=2)
