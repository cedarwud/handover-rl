"""
RL Environments - Gymnasium-Compatible Satellite Handover Environments

This module provides the production environment implementation for satellite
handover decision-making.

Components:
    SatelliteHandoverEnv: Online RL environment using real-time orbit-engine adapter

Features:
- Online RL mode with orbit calculations (precompute or realtime)
- Integration with AdapterWrapper (auto-selects optimal backend)
- Gymnasium API compatibility
- Multi-satellite state representation (Graph RL methodology)

Usage:
    from src.environments import SatelliteHandoverEnv
    from adapters import AdapterWrapper

    # Initialize adapter (auto-selects precompute or realtime backend)
    adapter = AdapterWrapper(config)

    # Create environment
    env = SatelliteHandoverEnv(adapter, satellite_ids, config)

    # Standard Gymnasium interface
    state, info = env.reset()
    action = agent.select_action(state)
    next_state, reward, terminated, truncated, info = env.step(action)

Note:
    - Current implementation: RVT-based reward following IEEE TAES 2024 paper
    - Older versions (V1, V2, V6) have been archived
"""

# RVT-based reward (RECOMMENDED - latest, academic standard)
from .satellite_handover_env import SatelliteHandoverEnv

__all__ = ['SatelliteHandoverEnv']

__version__ = "1.0.0"  # RVT-based reward (IEEE TAES 2024)
