#!/usr/bin/env python3
"""
Strongest RSRP Handover Strategy

最簡單的 handover 啟發式策略：總是選擇 RSRP 最強的衛星。

Academic Purpose:
- Baseline lower bound for demonstrating RL improvements
- Simplest possible strategy (no hysteresis, no thresholds)
- Expected to have high handover rate and ping-pong

Academic Compliance:
- No mock data or simplified models
- Uses real RSRP values from complete physics (ITU-R + 3GPP)
- Serves as comparison baseline for more sophisticated methods

Expected Performance (from orbit-engine analysis):
- Handover rate: 8-10% (high due to no hysteresis)
- Ping-pong rate: 10-15% (frequent back-and-forth switching)
- Average RSRP: Good (always selects strongest)
- Overall reward: Poor (excessive handovers penalized)
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class StrongestRSRPStrategy:
    """
    Strongest RSRP Handover Strategy

    Decision Rule:
        1. Find satellite with highest RSRP
        2. If strongest is not current serving satellite → handover
        3. If strongest is current serving satellite → stay

    No Parameters:
        - No threshold (always select strongest)
        - No hysteresis (no ping-pong protection)
        - No time-to-trigger (immediate switching)

    Use Case:
        - Baseline lower bound for comparison
        - Demonstrates why simple heuristics are insufficient
        - Shows value of more sophisticated strategies (A4, D2, RL)

    Expected Behavior:
        - ✅ Good RSRP (always connected to strongest signal)
        - ❌ High handover count (switches frequently)
        - ❌ High ping-pong (no hysteresis)
        - ❌ Poor overall performance (handover penalties)

    Example:
        >>> strategy = StrongestRSRPStrategy()
        >>> obs = env.reset()  # Shape: (10, 12)
        >>> action = strategy.select_action(obs, serving_satellite_idx=0)
        >>> # Returns 0 (stay) if satellite 0 has strongest RSRP
        >>> # Returns 1-10 (handover) if another satellite is stronger
    """

    def __init__(self):
        """
        Initialize Strongest RSRP strategy.

        No parameters needed (parameter-free strategy).
        """
        logger.info("StrongestRSRPStrategy initialized (parameter-free)")

    def select_action(
        self,
        observation: np.ndarray,
        serving_satellite_idx: Optional[int] = None,
        **kwargs
    ) -> int:
        """
        Select action based on strongest RSRP.

        Args:
            observation: (K, 12) satellite observation array
                Index 0: RSRP (dBm) - Reference Signal Received Power
                Index 10: Serving indicator (0/1) - Is current serving satellite

            serving_satellite_idx: Current serving satellite index (0-based)
                If None, uses observation[:, 10] to find serving satellite

            **kwargs: Ignored (for compatibility with unified evaluation)

        Returns:
            action: int
                0 = Stay (strongest is current serving)
                1-K = Handover to satellite at index (action - 1)

        Implementation Notes:
            - Complexity: O(K) - single pass to find argmax
            - No state memory (stateless strategy)
            - Deterministic (same observation → same action)

        Example:
            >>> obs = np.array([
            ...     [-85.0, ...],  # Satellite 0: RSRP = -85 dBm
            ...     [-90.0, ...],  # Satellite 1: RSRP = -90 dBm
            ...     [-80.0, ...],  # Satellite 2: RSRP = -80 dBm (strongest)
            ... ])
            >>> action = strategy.select_action(obs, serving_satellite_idx=0)
            >>> # Returns 3 (handover to satellite 2, which is at index 2)
        """
        # Extract RSRP column (index 0)
        rsrp_values = observation[:, 0]

        # Find satellite with strongest RSRP
        best_idx = int(np.argmax(rsrp_values))

        # Determine current serving satellite
        if serving_satellite_idx is None:
            # Use serving indicator from observation (index 10)
            serving_indicator = observation[:, 10]
            serving_satellites = np.where(serving_indicator == 1)[0]

            if len(serving_satellites) > 0:
                serving_satellite_idx = int(serving_satellites[0])
            else:
                # No current serving satellite (initial state)
                serving_satellite_idx = -1

        # Decision logic
        if best_idx == serving_satellite_idx:
            # Best satellite is already serving → stay
            return 0
        else:
            # Different satellite has stronger RSRP → handover
            # Action encoding: 1-K maps to satellite index 0-(K-1)
            return int(best_idx + 1)

    def get_config(self) -> dict:
        """
        Get strategy configuration.

        Returns:
            config: Dictionary with strategy metadata

        Example:
            >>> config = strategy.get_config()
            >>> print(config['name'])
            Strongest RSRP
        """
        return {
            'name': 'Strongest RSRP',
            'type': 'heuristic',
            'description': 'Always select satellite with highest RSRP',
            'parameters': {},  # No parameters
            'complexity': 'O(K)',
            'expected_ho_rate': '8-10%',
            'expected_ping_pong': '10-15%',
            'use_case': 'Baseline lower bound for comparison',
        }

    def __repr__(self) -> str:
        """String representation"""
        return "StrongestRSRPStrategy()"


# ========== Module Info ==========

__all__ = ['StrongestRSRPStrategy']

if __name__ == '__main__':
    # Test strategy
    print("=" * 60)
    print("Strongest RSRP Strategy")
    print("=" * 60)

    strategy = StrongestRSRPStrategy()
    print(f"\nStrategy: {strategy}")

    # Test case: 3 satellites
    print("\nTest Case: 3 satellites")
    obs = np.array([
        [-85.0, 0, 0, 1000, 45, 0, 0, 0, 0, 0, 1, 300],  # Sat 0: -85 dBm (serving)
        [-90.0, 0, 0, 1100, 40, 0, 0, 0, 0, 0, 0, 200],  # Sat 1: -90 dBm
        [-80.0, 0, 0, 900, 50, 0, 0, 0, 0, 0, 0, 400],   # Sat 2: -80 dBm (strongest)
    ], dtype=np.float32)

    action = strategy.select_action(obs, serving_satellite_idx=0)
    print(f"  RSRP values: {obs[:, 0]}")
    print(f"  Serving: Satellite 0 (-85 dBm)")
    print(f"  Strongest: Satellite 2 (-80 dBm)")
    print(f"  Decision: action={action} ", end="")

    if action == 0:
        print("(Stay)")
    else:
        print(f"(Handover to satellite {action - 1})")

    # Expected: action=3 (handover to satellite 2)
    assert action == 3, f"Expected action=3, got {action}"

    print("\n✅ Strategy test passed")
    print(f"\nConfig: {strategy.get_config()}")
