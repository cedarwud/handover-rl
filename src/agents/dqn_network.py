#!/usr/bin/env python3
"""
DQN Neural Network Architecture

Deep Q-Network architecture for satellite handover decision-making.

Network Structure:
    Input: 12-dimensional state
    Hidden: [128, 128] (2 hidden layers with ReLU activation)
    Output: 2-dimensional Q-values (maintain, handover)

SOURCE:
- Mnih et al. (2015) "Human-level control through deep reinforcement learning"
  Nature 518(7540): 529-533
- Network architecture adapted for satellite handover (12-dim continuous state)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for Handover Decision.

    Architecture:
        state (12) → FC(128) → ReLU → FC(128) → ReLU → FC(2) → Q-values

    Usage:
        network = DQNNetwork(state_dim=12, action_dim=2, hidden_dims=[128, 128])
        q_values = network(state_tensor)
    """

    def __init__(self,
                 state_dim: int = 12,
                 action_dim: int = 2,
                 hidden_dims: list = None,
                 dropout: float = 0.0):
        """
        Initialize DQN Network.

        Args:
            state_dim: Dimension of state space (default: 12)
            action_dim: Dimension of action space (default: 2)
            hidden_dims: List of hidden layer dimensions (default: [128, 128])
            dropout: Dropout rate (default: 0.0, disabled)
        """
        super(DQNNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims or [128, 128]
        self.dropout = dropout

        # Build network layers
        layers = []
        in_dim = state_dim

        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())

            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))

            in_dim = hidden_dim

        # Output layer (no activation - raw Q-values)
        layers.append(nn.Linear(in_dim, action_dim))

        # Sequential model
        self.network = nn.Sequential(*layers)

        # Initialize weights
        # SOURCE: Xavier initialization (Glorot & Bengio 2010)
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using Xavier initialization.

        SOURCE: Glorot & Bengio (2010) "Understanding the difficulty of
                training deep feedforward neural networks"
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            state: State tensor (batch_size, state_dim) or (state_dim,)

        Returns:
            q_values: Q-value tensor (batch_size, action_dim) or (action_dim,)
        """
        # Ensure input is 2D (add batch dimension if needed)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Forward pass
        q_values = self.network(state)

        # Remove batch dimension if input was 1D
        if squeeze_output:
            q_values = q_values.squeeze(0)

        return q_values

    def get_action(self, state: Union[np.ndarray, torch.Tensor],
                   deterministic: bool = False) -> int:
        """
        Get action from Q-values (greedy selection).

        Args:
            state: State (numpy array or tensor)
            deterministic: If True, always select max Q-value action

        Returns:
            action: Selected action (0 or 1)
        """
        # Convert to tensor if numpy
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)

        # Get Q-values
        with torch.no_grad():
            q_values = self.forward(state)

        # Select action with max Q-value
        action = q_values.argmax().item()

        return action

    def get_q_value(self, state: Union[np.ndarray, torch.Tensor],
                    action: int) -> float:
        """
        Get Q-value for a specific state-action pair.

        Args:
            state: State (numpy array or tensor)
            action: Action (0 or 1)

        Returns:
            q_value: Q-value for the state-action pair
        """
        # Convert to tensor if numpy
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)

        # Get Q-values
        with torch.no_grad():
            q_values = self.forward(state)

        # Get Q-value for specific action
        q_value = q_values[action].item()

        return q_value


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN Network (optional advanced version).

    Separates value function V(s) and advantage function A(s,a):
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))

    SOURCE: Wang et al. (2016) "Dueling Network Architectures for
            Deep Reinforcement Learning", ICML
    """

    def __init__(self,
                 state_dim: int = 12,
                 action_dim: int = 2,
                 hidden_dims: list = None,
                 dropout: float = 0.0):
        """
        Initialize Dueling DQN Network.

        Args:
            state_dim: Dimension of state space (default: 12)
            action_dim: Dimension of action space (default: 2)
            hidden_dims: List of hidden layer dimensions (default: [128, 128])
            dropout: Dropout rate (default: 0.0)
        """
        super(DuelingDQNNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims or [128, 128]
        self.dropout = dropout

        # Shared feature extractor
        feature_layers = []
        in_dim = state_dim

        for hidden_dim in self.hidden_dims[:-1]:  # All but last
            feature_layers.append(nn.Linear(in_dim, hidden_dim))
            feature_layers.append(nn.ReLU())
            if self.dropout > 0:
                feature_layers.append(nn.Dropout(self.dropout))
            in_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*feature_layers)
        feature_dim = in_dim

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[-1], 1)  # Single value output
        )

        # Advantage stream: A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[-1], action_dim)  # One per action
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dueling architecture.

        Q(s,a) = V(s) + (A(s,a) - mean_a'(A(s,a')))

        Args:
            state: State tensor (batch_size, state_dim) or (state_dim,)

        Returns:
            q_values: Q-value tensor (batch_size, action_dim) or (action_dim,)
        """
        # Ensure input is 2D
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Shared features
        features = self.feature_extractor(state)

        # Value and advantage
        value = self.value_stream(features)  # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, action_dim)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # Subtract mean advantage for stability
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        # Remove batch dimension if needed
        if squeeze_output:
            q_values = q_values.squeeze(0)

        return q_values

    def get_action(self, state: Union[np.ndarray, torch.Tensor],
                   deterministic: bool = False) -> int:
        """Get action from Q-values."""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)

        with torch.no_grad():
            q_values = self.forward(state)

        action = q_values.argmax().item()
        return action


# Example usage
if __name__ == "__main__":
    print("DQN Network Architecture")
    print("=" * 60)

    # Standard DQN
    print("\n1. Standard DQN Network:")
    dqn = DQNNetwork(state_dim=12, action_dim=2, hidden_dims=[128, 128])
    print(dqn)
    print(f"   Total parameters: {sum(p.numel() for p in dqn.parameters())}")

    # Test forward pass
    test_state = torch.randn(12)
    q_values = dqn(test_state)
    print(f"   Input shape: {test_state.shape}")
    print(f"   Output shape: {q_values.shape}")
    print(f"   Q-values: {q_values.detach().numpy()}")

    # Test batch
    test_batch = torch.randn(32, 12)
    q_batch = dqn(test_batch)
    print(f"   Batch input: {test_batch.shape}")
    print(f"   Batch output: {q_batch.shape}")

    # Test action selection
    action = dqn.get_action(test_state.numpy())
    print(f"   Selected action: {action}")

    # Dueling DQN
    print("\n2. Dueling DQN Network:")
    dueling_dqn = DuelingDQNNetwork(state_dim=12, action_dim=2, hidden_dims=[128, 128])
    print(f"   Total parameters: {sum(p.numel() for p in dueling_dqn.parameters())}")

    q_dueling = dueling_dqn(test_state)
    print(f"   Q-values (dueling): {q_dueling.detach().numpy()}")

    print("\n" + "=" * 60)
    print("✅ DQN Network verified!")
