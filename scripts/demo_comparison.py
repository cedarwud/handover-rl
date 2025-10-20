#!/usr/bin/env python3
"""
Quick Demonstration of Baseline Comparison

Uses Level 0 (10 satellites, 10 episodes) for quick demonstration.
This validates the evaluation framework works correctly.

For full comparison, use:
    ./scripts/run_level1_comparison.sh
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import yaml
import logging
import pandas as pd
from datetime import datetime

from adapters.orbit_engine_adapter import OrbitEngineAdapter
from environments.satellite_handover_env import SatelliteHandoverEnv
from strategies import StrongestRSRPStrategy, A4BasedStrategy, D2BasedStrategy
from utils.satellite_utils import load_stage4_optimized_satellites
from configs import get_level_config

# Import evaluation functions
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_strategies import compare_strategies

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("\n" + "=" * 80)
print("BASELINE COMPARISON DEMONSTRATION")
print("=" * 80)
print("\nThis is a quick demo using Level 0 (10 satellites, 10 episodes).")
print("For full Level 1 comparison (100 episodes), run:")
print("  ./scripts/run_level1_comparison.sh")
print("=" * 80 + "\n")

# Load config
config_path = Path(__file__).parent.parent / 'config' / 'data_gen_config.yaml'
logger.info(f"Loading config from: {config_path}")
with open(config_path) as f:
    config = yaml.safe_load(f)

# Use Level 0 for quick demo
level_config = get_level_config(0)
logger.info(f"\nDemo Configuration:")
logger.info(f"  Level: 0 (Smoke Test)")
logger.info(f"  Satellites: {level_config['num_satellites']}")
logger.info(f"  Episodes: {level_config['num_episodes']}")
logger.info(f"  Expected Time: ~{level_config['estimated_time_minutes']} minutes")

# Initialize adapter
logger.info("\nInitializing OrbitEngineAdapter...")
adapter = OrbitEngineAdapter(config)

# Load satellites
logger.info("Loading satellite pool...")
satellite_ids, metadata = load_stage4_optimized_satellites(
    constellation_filter='starlink',
    return_metadata=True
)
satellite_ids = satellite_ids[:level_config['num_satellites']]
logger.info(f"  Using {len(satellite_ids)} satellites")

# Create environment
logger.info("Creating environment...")
env = SatelliteHandoverEnv(adapter, satellite_ids, config)
logger.info("✅ Environment ready")

# Create strategies
logger.info("\nCreating strategies...")
strategies = {
    'Strongest RSRP': StrongestRSRPStrategy(),
    'A4-based': A4BasedStrategy(threshold_dbm=-100.0, hysteresis_db=1.5),
    'D2-based': D2BasedStrategy(threshold1_km=1412.8, threshold2_km=1005.8),
}
logger.info(f"  Created {len(strategies)} strategies")

# Run comparison
logger.info("\nRunning comparison...")
output_path = Path('results') / 'demo_comparison.csv'
df = compare_strategies(
    strategies,
    env,
    num_episodes=level_config['num_episodes'],
    seed=42,
    logger=logger,
    output_path=output_path
)

# Print results
print("\n" + "=" * 80)
print("COMPARISON RESULTS (Level 0 Demo)")
print("=" * 80 + "\n")
print(df.to_string(index=False))

# Analysis
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

best = df.iloc[0]
worst = df.iloc[-1]

print(f"\n🏆 Best Strategy: {best['strategy']}")
print(f"   Avg Reward: {best['avg_reward']:+.2f} ± {best['std_reward']:.2f}")
print(f"   Handover Rate: {best['handover_rate_pct']:.2f}%")
print(f"   Ping-pong Rate: {best['ping_pong_rate_pct']:.2f}%")
print(f"   Avg RSRP: {best['avg_rsrp_dbm']:.1f} dBm")

print(f"\n📊 Worst Strategy: {worst['strategy']}")
print(f"   Avg Reward: {worst['avg_reward']:+.2f} ± {worst['std_reward']:.2f}")
print(f"   Handover Rate: {worst['handover_rate_pct']:.2f}%")
print(f"   Ping-pong Rate: {worst['ping_pong_rate_pct']:.2f}%")

print(f"\n📈 Performance Gap:")
print(f"   Reward difference: {best['avg_reward'] - worst['avg_reward']:+.2f}")
print(f"   Ping-pong improvement: {worst['ping_pong_rate_pct'] - best['ping_pong_rate_pct']:.2f}% reduction")

print("\n" + "=" * 80)
print("✅ DEMONSTRATION COMPLETE")
print("=" * 80)
print("\nKey Findings (Level 0 - Quick Demo):")
print("  1. Evaluation framework working correctly")
print("  2. All 3 strategies successfully evaluated")
print("  3. Metrics collected and compared")
print("")
print("Expected Pattern (to be validated with Level 1):")
print("  - D2-based: Best rule-based (geometry-aware)")
print("  - A4-based: Standard 3GPP baseline")
print("  - Strongest RSRP: Simple heuristic (lower bound)")
print("")
print("Next Steps:")
print("  1. Run full Level 1 comparison (100 episodes):")
print("     ./scripts/run_level1_comparison.sh")
print("  2. Train DQN on Level 1")
print("  3. Compare RL vs rule-based baselines")
print("")
print(f"Demo results saved to: {output_path}")
print("=" * 80 + "\n")
