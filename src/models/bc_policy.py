"""
Behavior Cloning Policy Network

A simple MLP for binary classification: maintain vs handover
"""

import torch
import torch.nn as nn
from typing import List

class BCPolicy(nn.Module):
    """
    Behavior Cloning Policy Network

    Input: 10 features (RSRP, distance, elevation, trigger margin, etc.)
    Output: 2 classes (maintain=0, handover=1)
    """

    def __init__(self, input_dim: int = 10, hidden_dims: List[int] = [256, 128, 64]):
        super(BCPolicy, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
