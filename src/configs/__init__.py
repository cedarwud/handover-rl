"""
Configuration Module

Provides configuration utilities for the satellite handover RL framework.

Components:
    training_levels: Multi-level training strategy configuration (Novel Aspect #1)

Usage:
    from src.configs import get_level_config, TRAINING_LEVELS

    config = get_level_config(1)  # Get Level 1 configuration
"""

from .training_levels import (
    TRAINING_LEVELS,
    get_level_config,
    list_all_levels,
    validate_level_config
)

__all__ = [
    'TRAINING_LEVELS',
    'get_level_config',
    'list_all_levels',
    'validate_level_config',
]
