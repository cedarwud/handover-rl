#!/usr/bin/env python3
"""
Test Action Masking Implementation

This script verifies that the action masking mechanism works correctly:
1. Environment returns valid action_mask in info
2. Agent only selects valid actions
3. No invalid action warnings appear in logs
"""

import sys
import yaml
import logging
from pathlib import Path
from datetime import datetime

# Set up paths
sys.path.insert(0, str(Path(__file__).parent / "src"))

from environments.satellite_handover_env import SatelliteHandoverEnv
from agents import DQNAgent
from adapters import OrbitEngineAdapter

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def test_action_masking():
    """Test action masking with actual environment"""

    logger.info("=" * 60)
    logger.info("Testing Action Masking Implementation")
    logger.info("=" * 60)

    # Load config
    config_path = Path("config/diagnostic_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize adapter
    logger.info("\n1. Initializing OrbitEngineAdapter...")
    adapter = OrbitEngineAdapter(config=config)

    # Load satellites
    logger.info("2. Loading satellites...")
    from utils.satellite_utils import load_stage4_optimized_satellites
    satellite_ids = load_stage4_optimized_satellites()
    logger.info(f"   Loaded {len(satellite_ids)} satellites")

    # Create environment
    logger.info("\n3. Creating environment...")
    env = SatelliteHandoverEnv(
        adapter=adapter,
        satellite_ids=satellite_ids,
        config=config
    )
    logger.info(f"   Action space: {env.action_space}")
    logger.info(f"   Observation space: {env.observation_space}")

    # Create agent
    logger.info("\n4. Creating DQN agent...")
    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        config=config
    )

    # Test multiple episodes
    logger.info("\n5. Testing action masking over 10 steps...")
    obs, info = env.reset(seed=42)

    invalid_action_count = 0
    valid_action_count = 0

    for step in range(10):
        # Get action mask
        action_mask = info.get('action_mask')
        num_visible = info.get('num_visible', 0)

        logger.info(f"\n   Step {step + 1}:")
        logger.info(f"     Visible satellites: {num_visible}")
        logger.info(f"     Action mask: {action_mask}")
        logger.info(f"     Valid actions: {list(range(num_visible + 1))}")

        # Select action with masking
        action = agent.select_action(obs, deterministic=False, action_mask=action_mask)
        logger.info(f"     Selected action: {action}")

        # Verify action is valid
        if action_mask[action]:
            valid_action_count += 1
            logger.info(f"     ✅ Action {action} is VALID")
        else:
            invalid_action_count += 1
            logger.error(f"     ❌ Action {action} is INVALID!")

        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            logger.info("     Episode ended")
            break

    # Results
    logger.info("\n" + "=" * 60)
    logger.info("Test Results:")
    logger.info("=" * 60)
    logger.info(f"Valid actions selected: {valid_action_count}")
    logger.info(f"Invalid actions selected: {invalid_action_count}")

    if invalid_action_count == 0:
        logger.info("\n✅ SUCCESS: Action masking works correctly!")
        logger.info("   Agent only selected valid actions.")
        return True
    else:
        logger.error("\n❌ FAILURE: Action masking is not working!")
        logger.error(f"   Agent selected {invalid_action_count} invalid actions.")
        return False


if __name__ == "__main__":
    success = test_action_masking()
    sys.exit(0 if success else 1)
