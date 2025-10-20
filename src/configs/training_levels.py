#!/usr/bin/env python3
"""
Multi-Level Training Strategy Configuration

⭐ CRITICAL: Novel Aspect #1 - Must be preserved during refactoring ⭐

This module defines the 6-level progressive training strategy that enables
rapid development iteration without running 35-hour training every time.

Scientific Rationale:
    - Avoid 35-hour training for every experiment iteration
    - Progressive validation: 10 minutes → 35 hours
    - Fast idea validation (Level 1: 2h) vs full training (Level 5: 35h)
    - Reproducible experiments at each level

Research Contribution:
    - This is a Novel Aspect listed in README.md
    - Demonstrates efficient RL experimentation methodology
    - Other researchers can replicate this approach

Usage:
    from src.configs.training_levels import get_level_config, TRAINING_LEVELS

    # Get specific level
    config = get_level_config(3)  # Validation level
    print(f"Training {config['num_episodes']} episodes with {config['num_satellites']} satellites")

    # Iterate all levels
    for level_id, level_config in TRAINING_LEVELS.items():
        print(f"Level {level_id}: {level_config['name']}")

Integration with train.py:
    python train.py --algorithm dqn --level 0  # Smoke test
    python train.py --algorithm dqn --level 1  # Quick validation (recommended)
    python train.py --algorithm dqn --level 3  # Paper draft
    python train.py --algorithm dqn --level 5  # Final experiments
"""

from typing import Dict, Any


# ========== Training Level Definitions ==========

TRAINING_LEVELS: Dict[int, Dict[str, Any]] = {
    # Level 0: Smoke Test (10 minutes)
    # Use Case: System verification - ensure code runs without errors
    0: {
        'name': 'Smoke Test',
        'num_satellites': 10,
        'num_episodes': 10,
        'estimated_time_minutes': 10,
        'estimated_time_hours': 10 / 60,
        'description': 'Quick system verification - code runs without errors',
        'use_case': 'Debug, deployment verification, CI/CD testing',
        'overlap': 0.5,  # 50% overlap between episodes
        'log_interval': 1,  # Log every episode
        'checkpoint_interval': 5,  # Save checkpoint every 5 episodes
        'recommended': False,
    },

    # Level 1: Quick Validation (2 hours) ⭐ Recommended starting point
    # Use Case: Fast idea validation, hyperparameter testing
    1: {
        'name': 'Quick Validation',
        'num_satellites': 20,
        'num_episodes': 100,
        'estimated_time_minutes': 120,
        'estimated_time_hours': 2.0,
        'description': 'Verify training logic, observe learning curve',
        'use_case': 'Fast idea validation, hyperparameter testing, algorithm comparison',
        'overlap': 0.5,
        'log_interval': 10,
        'checkpoint_interval': 50,
        'recommended': True,  # ⭐ Recommended starting point
    },

    # Level 2: Development (6 hours)
    # Use Case: Development iteration, debugging
    2: {
        'name': 'Development',
        'num_satellites': 50,
        'num_episodes': 300,
        'estimated_time_minutes': 360,
        'estimated_time_hours': 6.0,
        'description': 'Debug hyperparameters and reward functions',
        'use_case': 'Development iteration, reward shaping, debugging',
        'overlap': 0.5,
        'log_interval': 10,
        'checkpoint_interval': 100,
        'recommended': False,
    },

    # Level 3: Validation (10 hours)
    # Use Case: Validate effectiveness, paper draft experiments
    3: {
        'name': 'Validation',
        'num_satellites': 101,  # Full Starlink pool
        'num_episodes': 500,
        'estimated_time_minutes': 600,
        'estimated_time_hours': 10.0,
        'description': 'Validate effectiveness with full satellite pool',
        'use_case': 'Paper draft experiments, baseline validation',
        'overlap': 0.5,
        'log_interval': 10,
        'checkpoint_interval': 100,
        'recommended': False,
    },

    # Level 4: Baseline (21 hours)
    # Use Case: Establish stable baseline for paper experiments
    4: {
        'name': 'Baseline',
        'num_satellites': 101,
        'num_episodes': 1000,
        'estimated_time_minutes': 1260,
        'estimated_time_hours': 21.0,
        'description': 'Establish stable baseline for comparisons',
        'use_case': 'Paper experiments, baseline establishment',
        'overlap': 0.5,
        'log_interval': 10,
        'checkpoint_interval': 100,
        'recommended': False,
    },

    # Level 5: Full Training (35 hours)
    # Use Case: Final paper experiments, publication-quality results
    5: {
        'name': 'Full Training',
        'num_satellites': 101,
        'num_episodes': 1700,
        'estimated_time_minutes': 2100,
        'estimated_time_hours': 35.0,
        'description': 'Complete training for publication-quality results',
        'use_case': 'Final paper experiments, publication results',
        'overlap': 0.5,
        'log_interval': 10,
        'checkpoint_interval': 100,
        'recommended': False,
    },
}


# ========== Helper Functions ==========

def get_level_config(level: int) -> Dict[str, Any]:
    """
    Get configuration for specific training level

    Args:
        level: Training level (0-5)

    Returns:
        config: Dictionary containing level configuration

    Raises:
        ValueError: If level is not in range 0-5

    Example:
        >>> config = get_level_config(1)
        >>> print(f"Level 1: {config['name']}")
        Level 1: Quick Validation
        >>> print(f"Episodes: {config['num_episodes']}")
        Episodes: 100
        >>> print(f"Time: {config['estimated_time_hours']}h")
        Time: 2.0h
    """
    if level not in TRAINING_LEVELS:
        raise ValueError(
            f"Invalid training level: {level}. Must be 0-5.\n"
            f"Available levels:\n"
            + "\n".join([
                f"  Level {lvl}: {cfg['name']} ({cfg['estimated_time_hours']}h, "
                f"{cfg['num_episodes']} episodes)"
                for lvl, cfg in TRAINING_LEVELS.items()
            ])
        )

    return TRAINING_LEVELS[level].copy()


def list_all_levels() -> str:
    """
    Get formatted string of all training levels

    Returns:
        levels_str: Human-readable string describing all levels

    Example:
        >>> print(list_all_levels())
        Multi-Level Training Strategy
        ==============================
        Level 0: Smoke Test (10 min, 10 episodes, 10 satellites)
        ...
    """
    lines = ["Multi-Level Training Strategy", "=" * 60]

    for level_id in sorted(TRAINING_LEVELS.keys()):
        config = TRAINING_LEVELS[level_id]
        recommended = " ⭐ RECOMMENDED" if config.get('recommended', False) else ""

        # Format time
        if config['estimated_time_hours'] < 1:
            time_str = f"{config['estimated_time_minutes']} min"
        else:
            time_str = f"{config['estimated_time_hours']:.1f}h"

        lines.append(
            f"Level {level_id}: {config['name']} "
            f"({time_str}, {config['num_episodes']} episodes, "
            f"{config['num_satellites']} satellites){recommended}"
        )
        lines.append(f"  Use Case: {config['use_case']}")

    return "\n".join(lines)


def validate_level_config(level: int) -> bool:
    """
    Validate that a level configuration is complete and consistent

    Args:
        level: Training level to validate

    Returns:
        is_valid: True if configuration is valid

    Raises:
        AssertionError: If configuration has missing or invalid fields
    """
    config = get_level_config(level)

    # Required fields
    required_fields = [
        'name', 'num_satellites', 'num_episodes',
        'estimated_time_minutes', 'estimated_time_hours',
        'description', 'use_case', 'overlap',
        'log_interval', 'checkpoint_interval'
    ]

    for field in required_fields:
        assert field in config, f"Level {level} missing field: {field}"

    # Validation rules
    assert config['num_satellites'] > 0, "num_satellites must be positive"
    assert config['num_episodes'] > 0, "num_episodes must be positive"
    assert 0 <= config['overlap'] <= 1, "overlap must be in [0, 1]"
    assert config['log_interval'] > 0, "log_interval must be positive"
    assert config['checkpoint_interval'] > 0, "checkpoint_interval must be positive"

    # Time consistency
    expected_hours = config['estimated_time_minutes'] / 60
    assert abs(expected_hours - config['estimated_time_hours']) < 0.01, \
        "Time estimates inconsistent"

    return True


# ========== Validation on Import ==========

# Validate all levels on module import
for level_id in TRAINING_LEVELS.keys():
    try:
        validate_level_config(level_id)
    except AssertionError as e:
        raise ValueError(f"Invalid configuration for Level {level_id}: {e}")


# ========== Module Info ==========

if __name__ == '__main__':
    # Print all levels when run directly
    print(list_all_levels())
    print("\n" + "=" * 60)
    print("✅ All training levels validated successfully")
    print("\nRecommended starting point:")
    recommended = get_level_config(1)
    print(f"  Level 1: {recommended['name']}")
    print(f"  Time: {recommended['estimated_time_hours']}h")
    print(f"  Episodes: {recommended['num_episodes']}")
    print(f"  Use case: {recommended['use_case']}")
