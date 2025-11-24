"""
RL Environments - Gymnasium-Compatible Satellite Handover Environments

This module provides concrete environment implementations for satellite
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
    - HandoverEnvironment (offline RL) has been deprecated and moved to archive/
    - Current implementation uses SatelliteHandoverEnv (online RL)
"""

from .satellite_handover_env import SatelliteHandoverEnv

__all__ = ['SatelliteHandoverEnv']

__version__ = "3.0.0"  # Online RL version
