#!/usr/bin/env python3
"""
DQN Agent for Satellite Handover

Deep Q-Network agent implementation for LEO satellite handover optimization.

Features:
- ε-greedy exploration
- Experience replay
- Target network with soft/hard updates
- Gradient clipping
- Double DQN support (optional)

Algorithm:
    1. Observe state s
    2. Select action a using ε-greedy policy
    3. Execute action, observe reward r and next state s'
    4. Store transition (s, a, r, s', done) in replay buffer
    5. Sample random batch from buffer
    6. Compute target: y = r + γ * max_a' Q_target(s', a')
    7. Update Q-network to minimize (Q(s,a) - y)^2
    8. Periodically update target network

SOURCE:
- Mnih et al. (2015) "Human-level control through deep reinforcement learning"
  Nature 518(7540): 529-533
- Van Hasselt et al. (2016) "Deep Reinforcement Learning with Double Q-learning", AAAI
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Union
from pathlib import Path

# Add parent directory to path for imports
CURRENT_DIR = Path(__file__).parent
RL_CORE_DIR = CURRENT_DIR.parent / "rl_core"
sys.path.insert(0, str(RL_CORE_DIR))

from base_agent import BaseRLAgent
from .dqn_network import DQNNetwork, DuelingDQNNetwork
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent(BaseRLAgent):
    """
    DQN Agent for Satellite Handover.

    Implements Deep Q-Network with experience replay and target network.

    Usage:
        agent = DQNAgent(state_dim=12, action_dim=2, config)
        action = agent.select_action(state, eval_mode=False)
        loss = agent.update()
        agent.save('checkpoint.pth')
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: Dict):
        """
        Initialize DQN Agent.

        Args:
            state_dim: State space dimension (default: 12)
            action_dim: Action space dimension (default: 2)
            config: Configuration dictionary with:
                - learning_rate: Learning rate (default: 1e-4)
                - gamma: Discount factor (default: 0.99)
                - epsilon_start: Initial exploration rate (default: 1.0)
                - epsilon_end: Final exploration rate (default: 0.01)
                - epsilon_decay: Decay rate (default: 0.995)
                - batch_size: Batch size for training (default: 64)
                - buffer_size: Replay buffer capacity (default: 100000)
                - target_update_frequency: Target network update frequency (default: 1000)
                - tau: Soft update coefficient (default: 0.005)
                - use_double_dqn: Use Double DQN (default: False)
                - use_dueling: Use Dueling architecture (default: False)
                - use_prioritized_replay: Use prioritized replay (default: False)
        """
        super().__init__(state_dim, action_dim, config)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('batch_size', 64)
        self.buffer_size = config.get('buffer_size', 100000)
        self.target_update_frequency = config.get('target_update_frequency', 1000)
        self.tau = config.get('tau', 0.005)  # For soft updates
        self.use_soft_update = config.get('use_soft_update', False)
        self.gradient_clip = config.get('gradient_clip', 10.0)

        # Algorithm variants
        self.use_double_dqn = config.get('use_double_dqn', False)
        self.use_dueling = config.get('use_dueling', False)
        self.use_prioritized_replay = config.get('use_prioritized_replay', False)

        # Network architecture
        hidden_dims = config.get('hidden_dims', [128, 128])

        # Create Q-networks
        NetworkClass = DuelingDQNNetwork if self.use_dueling else DQNNetwork

        self.q_network = NetworkClass(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)

        self.target_network = NetworkClass(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)

        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Always in eval mode

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Loss function
        self.criterion = nn.MSELoss()

        # Replay buffer
        if self.use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=self.buffer_size,
                alpha=config.get('priority_alpha', 0.6),
                beta=config.get('priority_beta', 0.4)
            )
        else:
            self.replay_buffer = ReplayBuffer(capacity=self.buffer_size)

        # Training state
        self.update_counter = 0

        print(f"✅ DQNAgent initialized:")
        print(f"   Device: {self.device}")
        print(f"   Network: {'Dueling' if self.use_dueling else 'Standard'} DQN")
        print(f"   Double DQN: {self.use_double_dqn}")
        print(f"   Prioritized Replay: {self.use_prioritized_replay}")
        print(f"   Parameters: {sum(p.numel() for p in self.q_network.parameters())}")

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Select action using ε-greedy policy.

        Args:
            state: Current state (12-dim numpy array)
            eval_mode: If True, use greedy policy (no exploration)

        Returns:
            action: Selected action (0=maintain, 1=handover)
        """
        # Greedy action in eval mode
        if eval_mode:
            state_tensor = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
            return action

        # ε-greedy exploration
        if np.random.rand() < self.epsilon:
            # Random action
            action = np.random.randint(0, self.action_dim)
        else:
            # Greedy action
            state_tensor = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()

        return action

    def update(self) -> Union[float, Dict[str, float]]:
        """
        Update Q-network using experience replay.

        Returns:
            loss: Training loss (float or dict)
        """
        # Check if buffer has enough samples
        if not self.replay_buffer.is_ready(self.batch_size):
            return 0.0

        # Sample batch from replay buffer
        if self.use_prioritized_replay:
            states, actions, rewards, next_states, dones, indices, weights = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values: Q(s, a)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values: y = r + γ * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: Use online network to select action, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(1)[0]

            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss (weighted for prioritized replay)
        td_errors = current_q_values - target_q_values
        loss = (weights * td_errors.pow(2)).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)

        self.optimizer.step()

        # Update priorities if using prioritized replay
        if self.use_prioritized_replay:
            priorities = td_errors.abs().detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, priorities)

        # Update target network
        self.update_counter += 1
        if self.use_soft_update:
            # Soft update: θ_target = τ*θ_local + (1-τ)*θ_target
            self._soft_update_target_network()
        else:
            # Hard update: Copy weights every N steps
            if self.update_counter % self.target_update_frequency == 0:
                self._hard_update_target_network()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {
            'total_loss': loss.item(),
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer)
        }

    def _soft_update_target_network(self):
        """Soft update: θ_target = τ*θ_local + (1-τ)*θ_target"""
        for target_param, local_param in zip(self.target_network.parameters(),
                                             self.q_network.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def _hard_update_target_network(self):
        """Hard update: Copy all weights from Q-network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def on_step(self,
                state: np.ndarray,
                action: int,
                reward: float,
                next_state: np.ndarray,
                done: bool):
        """
        Store transition in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def save(self, path: str):
        """
        Save agent checkpoint.

        Args:
            path: Checkpoint file path
        """
        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'update_counter': self.update_counter,
            'config': self.config
        }

        torch.save(checkpoint, path)

    def load(self, path: str):
        """
        Load agent checkpoint.

        Args:
            path: Checkpoint file path
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']
        self.total_episodes = checkpoint['total_episodes']
        self.update_counter = checkpoint['update_counter']

    def get_training_state(self) -> Dict:
        """
        Get current training state.

        Returns:
            state: Dictionary with training state information
        """
        base_state = super().get_training_state()

        dqn_state = {
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'buffer_utilization': len(self.replay_buffer) / self.buffer_size,
            'update_counter': self.update_counter,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma
        }

        return {**base_state, **dqn_state}


# Example usage
if __name__ == "__main__":
    print("DQN Agent for Satellite Handover")
    print("=" * 60)

    # Configuration
    config = {
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 64,
        'buffer_size': 10000,
        'target_update_frequency': 100,
        'tau': 0.005,
        'use_soft_update': False,
        'use_double_dqn': True,
        'use_dueling': False,
        'use_prioritized_replay': False,
        'hidden_dims': [128, 128],
        'gradient_clip': 10.0
    }

    # Create agent
    agent = DQNAgent(state_dim=12, action_dim=2, config=config)

    print(f"\n📊 Agent configuration:")
    print(f"   State dim: {agent.state_dim}")
    print(f"   Action dim: {agent.action_dim}")
    print(f"   Epsilon: {agent.epsilon}")
    print(f"   Buffer capacity: {agent.buffer_size}")

    # Test action selection
    print("\n🎲 Testing action selection:")
    test_state = np.random.randn(12).astype(np.float32)

    # Exploration mode
    action_explore = agent.select_action(test_state, eval_mode=False)
    print(f"   Exploration mode: action={action_explore}")

    # Evaluation mode
    action_eval = agent.select_action(test_state, eval_mode=True)
    print(f"   Evaluation mode: action={action_eval}")

    # Test replay buffer
    print("\n💾 Testing replay buffer:")
    for i in range(100):
        state = np.random.randn(12).astype(np.float32)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(12).astype(np.float32)
        done = (i % 20 == 19)

        agent.on_step(state, action, reward, next_state, done)

    print(f"   Buffer size: {len(agent.replay_buffer)}")
    print(f"   Ready for training: {agent.replay_buffer.is_ready(agent.batch_size)}")

    # Test update
    print("\n🔄 Testing update:")
    if agent.replay_buffer.is_ready(agent.batch_size):
        loss_info = agent.update()
        print(f"   Loss: {loss_info['total_loss']:.4f}")
        print(f"   Epsilon: {loss_info['epsilon']:.4f}")
        print(f"   Buffer size: {loss_info['buffer_size']}")

    # Test save/load
    print("\n💾 Testing save/load:")
    checkpoint_path = "/tmp/dqn_test.pth"
    agent.save(checkpoint_path)
    print(f"   Checkpoint saved: {checkpoint_path}")

    # Create new agent and load
    new_agent = DQNAgent(state_dim=12, action_dim=2, config=config)
    new_agent.load(checkpoint_path)
    print(f"   Checkpoint loaded successfully")
    print(f"   Loaded epsilon: {new_agent.epsilon:.4f}")
    print(f"   Loaded steps: {new_agent.total_steps}")

    # Training state
    print("\n📈 Training state:")
    state = agent.get_training_state()
    for key, value in state.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 60)
    print("✅ DQN Agent verified!")
