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

# V9: RVT-based reward (has reward misalignment issues)
from .satellite_handover_env import SatelliteHandoverEnv

# V10: Connectivity-centric reward (RECOMMENDED - aligned with operational objectives)
from .satellite_handover_env_v10 import SatelliteHandoverEnv as SatelliteHandoverEnvV10

__all__ = ['SatelliteHandoverEnv', 'SatelliteHandoverEnvV10']

__version__ = "1.1.0"  # Added V10 connectivity-centric reward (2025-12-19)
