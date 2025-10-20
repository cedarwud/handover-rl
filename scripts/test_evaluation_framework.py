#!/usr/bin/env python3
"""
Quick test for evaluation framework

Tests the evaluate_strategies.py framework with minimal setup.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import yaml
import logging
from adapters.orbit_engine_adapter import OrbitEngineAdapter
from environments.satellite_handover_env import SatelliteHandoverEnv
from strategies import StrongestRSRPStrategy, A4BasedStrategy, D2BasedStrategy
from utils.satellite_utils import load_stage4_optimized_satellites
from configs import get_level_config

# Import evaluation function
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_strategies import evaluate_strategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 60)
print("Testing Evaluation Framework")
print("=" * 60)

# Load config
config_path = Path(__file__).parent.parent / 'config' / 'data_gen_config.yaml'
with open(config_path) as f:
    config = yaml.safe_load(f)

# Use Level 0 for quick test (10 satellites, 10 episodes)
level_config = get_level_config(0)
print(f"\nUsing Level 0 for quick test:")
print(f"  Satellites: {level_config['num_satellites']}")
print(f"  Episodes: {level_config['num_episodes']}")

# Initialize adapter
print("\nInitializing adapter...")
adapter = OrbitEngineAdapter(config)

# Load satellites
print("Loading satellites...")
satellite_ids, _ = load_stage4_optimized_satellites(
    constellation_filter='starlink',
    return_metadata=True
)
satellite_ids = satellite_ids[:level_config['num_satellites']]
print(f"  Using {len(satellite_ids)} satellites")

# Create environment
print("Creating environment...")
env = SatelliteHandoverEnv(adapter, satellite_ids, config)

# Create one strategy for testing
print("\nTesting with D2-based strategy...")
strategy = D2BasedStrategy(threshold1_km=1412.8, threshold2_km=1005.8)

# Evaluate
print("Running evaluation...")
metrics = evaluate_strategy(
    strategy,
    env,
    num_episodes=level_config['num_episodes'],
    seed=42,
    logger=logger
)

print("\n" + "=" * 60)
print("✅ Evaluation Framework Test Complete!")
print("=" * 60)
print(f"\nResults:")
print(f"  Strategy: {metrics['strategy_name']}")
print(f"  Avg Reward: {metrics['avg_reward']:+.2f} ± {metrics['std_reward']:.2f}")
print(f"  Avg Handovers: {metrics['avg_handovers']:.2f}")
print(f"  Ping-pong Rate: {metrics['ping_pong_rate_pct']:.2f}%")
print("\n✅ evaluate_strategy() function working correctly!")
