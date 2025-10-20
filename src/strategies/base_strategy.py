#!/usr/bin/env python3
"""
Base Strategy Protocol

定義 handover decision strategies 的統一介面。

Design Philosophy:
- ❌ NOT an abstract base class (不強制繼承)
- ✅ Protocol-based (使用 duck typing)
- ✅ 支持 RL agents 和 rule-based strategies
- ✅ 最小介面要求（只需要 select_action()）

This allows:
- RL agents (e.g., DQNAgent) to work without modification
- Rule-based strategies to be independent of RL framework
- Unified evaluation through duck typing

Academic Compliance:
- Rule-based strategies use official standards (3GPP TS 38.331)
- Parameters sourced from real data (orbit-engine)
- No simplified models or mock implementations
"""

from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class HandoverStrategy(Protocol):
    """
    Protocol for handover decision strategies.

    ANY class that implements select_action() can be used as a strategy,
    whether it's an RL agent or a rule-based method.

    This enables:
    1. RL agents (DQNAgent, PPOAgent) - no code changes needed
    2. Rule-based methods (A4-based, D2-based) - independent implementation
    3. Unified evaluation framework

    Example (RL Agent):
        >>> agent = DQNAgent(obs_space, action_space, config)
        >>> action = agent.select_action(observation, deterministic=True)
        >>> # Works because DQNAgent has select_action()

    Example (Rule-based Strategy):
        >>> strategy = A4BasedStrategy(threshold_dbm=-100.0)
        >>> action = strategy.select_action(observation)
        >>> # Works because A4BasedStrategy has select_action()

    Example (Unified Evaluation):
        >>> strategies = {
        ...     'DQN': dqn_agent,
        ...     'A4-based': a4_strategy,
        ...     'D2-based': d2_strategy,
        ... }
        >>> for name, strategy in strategies.items():
        ...     action = strategy.select_action(obs)  # Duck typing!
    """

    def select_action(self, observation: np.ndarray, **kwargs) -> int:
        """
        Select handover action based on observation.

        Args:
            observation: (K, 12) array from SatelliteHandoverEnv
                Index 0: RSRP (dBm) - Reference Signal Received Power
                Index 1: RSRQ (dB) - Reference Signal Received Quality
                Index 2: SINR (dB) - Signal-to-Interference-plus-Noise Ratio
                Index 3: Distance (km) - Satellite-to-ground distance
                Index 4: Elevation (deg) - Elevation angle
                Index 5: Azimuth (deg) - Azimuth angle
                Index 6: Doppler (Hz) - Doppler shift
                Index 7: Path Loss (dB) - Free-space + atmospheric loss
                Index 8: Atmospheric Loss (dB) - ITU-R P.676-13
                Index 9: Velocity (km/s) - Satellite velocity
                Index 10: Serving indicator (0/1) - Is current serving satellite
                Index 11: Visibility time (sec) - Remaining visibility duration

            **kwargs: Optional information for context-aware decisions
                serving_satellite_idx: int - Current serving satellite index (0-based)
                    Used by D2-based strategy to compare current vs neighbors
                deterministic: bool - Whether to use deterministic policy (for RL agents)

        Returns:
            action: int
                0 = Stay with current satellite (no handover)
                1 to K = Handover to satellite at index (action - 1)

        Implementation Notes:
        - RL agents may use 'deterministic' kwarg for evaluation vs training
        - Rule-based strategies typically ignore 'deterministic' (already deterministic)
        - serving_satellite_idx enables strategies to compare serving vs neighbors

        Example:
            >>> obs = env.reset()  # Shape: (10, 12)
            >>> # For RL agent
            >>> action = agent.select_action(obs, deterministic=True)
            >>> # For rule-based strategy
            >>> action = strategy.select_action(obs, serving_satellite_idx=0)
        """
        ...


# ========== Helper Functions ==========

def is_valid_strategy(obj) -> bool:
    """
    Check if object implements HandoverStrategy protocol.

    Args:
        obj: Object to check

    Returns:
        is_valid: True if object can be used as a strategy

    Example:
        >>> from agents import DQNAgent
        >>> from strategies import A4BasedStrategy
        >>> agent = DQNAgent(...)
        >>> strategy = A4BasedStrategy(...)
        >>> assert is_valid_strategy(agent)      # True (has select_action)
        >>> assert is_valid_strategy(strategy)   # True (has select_action)
        >>> assert not is_valid_strategy(None)   # False
    """
    return isinstance(obj, HandoverStrategy)


def validate_observation(observation: np.ndarray) -> bool:
    """
    Validate observation format for strategies.

    Args:
        observation: Observation array from environment

    Returns:
        is_valid: True if observation format is correct

    Raises:
        ValueError: If observation format is invalid

    Example:
        >>> obs = np.random.randn(10, 12)
        >>> validate_observation(obs)  # True
        >>> obs_bad = np.random.randn(10, 5)
        >>> validate_observation(obs_bad)  # ValueError
    """
    if not isinstance(observation, np.ndarray):
        raise ValueError(f"Observation must be numpy array, got {type(observation)}")

    if observation.ndim != 2:
        raise ValueError(f"Observation must be 2D array (K, 12), got shape {observation.shape}")

    if observation.shape[1] != 12:
        raise ValueError(f"Observation must have 12 features per satellite, got {observation.shape[1]}")

    return True


# ========== Module Info ==========

__all__ = [
    'HandoverStrategy',
    'is_valid_strategy',
    'validate_observation',
]

__version__ = "2.0.0-phase2"


if __name__ == '__main__':
    # Test protocol
    print("=" * 60)
    print("HandoverStrategy Protocol")
    print("=" * 60)
    print("\nProtocol definition:")
    print("  - select_action(observation, **kwargs) -> int")
    print("\nSupported strategies:")
    print("  ✓ RL agents (DQNAgent, PPOAgent, etc.)")
    print("  ✓ Rule-based methods (A4-based, D2-based, etc.)")
    print("\nUsage:")
    print("  action = strategy.select_action(observation)")
    print("\n✅ Protocol ready for Phase 2 implementation")
