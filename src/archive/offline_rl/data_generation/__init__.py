"""
Data Generation - Training Data Generators

This module provides tools for generating RL training episodes using
real satellite orbital data.

Components:
    RLDataGenerator: Main data generator using orbit-engine adapter
    EpisodeBuilder: Utilities for episode construction and validation

Features:
- 30-day dataset generation with TLE precision strategy
- Ground truth action labeling using future state lookahead
- Episode validation and quality filtering
- Statistics calculation and aggregation
- Train/val/test splitting

Usage:
    from src.data_generation import RLDataGenerator, EpisodeBuilder

    # Generate dataset
    generator = RLDataGenerator(config)
    generator.generate_dataset(start_date, end_date, output_dir)

    # Build and validate episodes
    builder = EpisodeBuilder(config)
    episode = builder.build_episode(states, actions, rewards, timestamps)
    is_valid, info = builder.validate_episode(episode)
"""

from .rl_data_generator import RLDataGenerator
from .episode_builder import EpisodeBuilder, load_episodes, create_dataset_splits

__all__ = [
    'RLDataGenerator',
    'EpisodeBuilder',
    'load_episodes',
    'create_dataset_splits'
]

__version__ = "2.0.0"
