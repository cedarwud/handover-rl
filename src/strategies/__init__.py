"""
Strategies Module - Rule-based Handover Methods

Rule-based handover strategies for baseline comparison.

Design Principles:
- Independent of RL framework (only depends on environment API)
- Based on official standards (3GPP TS 38.331)
- Parameters from real data (orbit-engine)
- Duck typing for unified evaluation

Available Strategies:
    StrongestRSRPStrategy: Simple heuristic (always select strongest RSRP)
    A4BasedStrategy: 3GPP A4 Event + RSRP selection
    D2BasedStrategy: 3GPP D2 Event + distance selection (NTN-specific)

Protocol:
    HandoverStrategy: Protocol for duck typing (any object with select_action())

Usage:
    from src.strategies import HandoverStrategy, A4BasedStrategy

    strategy = A4BasedStrategy(threshold_dbm=-100.0)
    action = strategy.select_action(observation)
"""

from .base_strategy import (
    HandoverStrategy,
    is_valid_strategy,
    validate_observation,
)

# Strategies (implemented in Task 2.2-2.4)
from .strongest_rsrp import StrongestRSRPStrategy
from .a4_based_strategy import A4BasedStrategy
from .d2_based_strategy import D2BasedStrategy

__all__ = [
    # Protocol
    'HandoverStrategy',
    'is_valid_strategy',
    'validate_observation',
    # Strategies
    'StrongestRSRPStrategy',
    'A4BasedStrategy',
    'D2BasedStrategy',
]

__version__ = "2.0.0-phase2"
