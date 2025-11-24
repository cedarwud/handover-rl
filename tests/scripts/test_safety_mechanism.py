#!/usr/bin/env python3
"""
Test Safety Mechanism - Episode 520-525

Tests the safety protection mechanisms:
1. Timeout protection (10 minutes)
2. Resource monitoring (CPU/RAM)
3. Exception handling (try-catch)
4. Auto-skip problematic episodes

Expected behavior:
- Episode 520-521: Should complete normally (~13s each)
- Episode 522: Should timeout after 600s or complete if fixed (~291s)
- Episode 523-525: Similar behavior to 522
- Training should NOT crash, continue to completion
"""

import sys
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from adapters import AdapterWrapper
from environments.satellite_handover_env import SatelliteHandoverEnv
from agents import DQNAgent
from trainers import OffPolicyTrainer
from utils.satellite_utils import load_stage4_optimized_satellites
from configs import get_level_config


def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    return logging.getLogger(__name__)


def main():
    logger = setup_logging()
    logger.info("="*80)
    logger.info("Safety Mechanism Test - Episodes 520-525")
    logger.info("="*80)

    # Load config
    config_path = Path(__file__).parent.parent / 'config' / 'diagnostic_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get level 4 config (has safety settings)
    level_config = get_level_config(4)

    # Merge safety settings
    if 'training' not in config:
        config['training'] = {}
    config['training'].update({
        'num_episodes': 6,  # Only test 520-525
        'episode_timeout_seconds': level_config.get('episode_timeout_seconds', 600),
        'max_memory_percent': level_config.get('max_memory_percent', 90),
        'max_cpu_percent': level_config.get('max_cpu_percent', 95),
        'enable_safety_checks': level_config.get('enable_safety_checks', True),
        'resource_check_interval': level_config.get('resource_check_interval', 10),
    })

    logger.info(f"Safety settings:")
    logger.info(f"  Timeout: {config['training']['episode_timeout_seconds']}s")
    logger.info(f"  Max memory: {config['training']['max_memory_percent']}%")
    logger.info(f"  Max CPU: {config['training']['max_cpu_percent']}%")

    # Initialize adapter
    logger.info("Initializing adapter...")
    adapter = AdapterWrapper(config)

    # Load satellites
    logger.info("Loading satellites...")
    satellite_ids = load_stage4_optimized_satellites()
    logger.info(f"  Loaded {len(satellite_ids)} satellites")

    # Create environment
    logger.info("Creating environment...")
    env = SatelliteHandoverEnv(adapter, satellite_ids, config)

    # Create agent
    logger.info("Creating DQN agent...")
    agent = DQNAgent(env.observation_space, env.action_space, config)

    # Create trainer
    logger.info("Creating trainer...")
    trainer = OffPolicyTrainer(env, agent, config)

    # Test episodes 520-525
    start_time_base = datetime(2025, 10, 10, 0, 0, 0)
    episode_stride_minutes = int(20 * 0.5)  # 50% overlap

    results = []

    logger.info("\n" + "="*80)
    logger.info("Starting Episode Tests")
    logger.info("="*80)

    for episode_idx in range(520, 526):
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing Episode {episode_idx}")
        logger.info(f"{'='*80}")

        # Calculate episode start time
        time_offset_minutes = episode_idx * episode_stride_minutes
        episode_start_time = start_time_base + timedelta(minutes=time_offset_minutes)

        # Train episode with safety protection
        test_start = datetime.now()
        metrics = trainer.train_episode(
            episode_idx=episode_idx,
            episode_start_time=episode_start_time,
            seed=42 + episode_idx
        )
        test_duration = (datetime.now() - test_start).total_seconds()

        # Record results
        result = {
            'episode': episode_idx,
            'skipped': metrics.get('skipped', False),
            'error': metrics.get('error', None),
            'duration_s': test_duration,
            'reward': metrics['reward'],
            'length': metrics['length'],
        }
        results.append(result)

        # Log result
        if result['skipped']:
            logger.warning(f"❌ Episode {episode_idx}: SKIPPED (duration={test_duration:.1f}s)")
            logger.warning(f"   Error: {result['error']}")
        else:
            logger.info(f"✅ Episode {episode_idx}: SUCCESS (duration={test_duration:.1f}s)")
            logger.info(f"   Reward: {result['reward']:.2f}, Length: {result['length']} steps")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("Test Summary")
    logger.info("="*80)

    total_episodes = len(results)
    skipped_episodes = sum(1 for r in results if r['skipped'])
    successful_episodes = total_episodes - skipped_episodes

    logger.info(f"Total episodes tested: {total_episodes}")
    logger.info(f"Successful episodes: {successful_episodes}")
    logger.info(f"Skipped episodes: {skipped_episodes}")
    logger.info(f"Success rate: {100.0 * successful_episodes / total_episodes:.1f}%")

    logger.info(f"\nDetailed Results:")
    logger.info(f"{'Episode':<10} {'Status':<12} {'Duration':<12} {'Error':<40}")
    logger.info(f"{'-'*74}")
    for r in results:
        status = "SKIPPED" if r['skipped'] else "SUCCESS"
        error = (r['error'][:37] + '...') if r['error'] and len(r['error']) > 40 else (r['error'] or '-')
        logger.info(f"{r['episode']:<10} {status:<12} {r['duration_s']:<12.1f} {error:<40}")

    # Check if Episode 522 was handled
    ep522 = [r for r in results if r['episode'] == 522][0]
    logger.info(f"\n{'='*80}")
    logger.info("Episode 522 Analysis (The Problem Episode)")
    logger.info("="*80)
    logger.info(f"Status: {'SKIPPED' if ep522['skipped'] else 'SUCCESS'}")
    logger.info(f"Duration: {ep522['duration_s']:.1f}s")
    if ep522['skipped']:
        logger.info(f"Error type: {ep522['error'].split(':')[0] if ep522['error'] else 'Unknown'}")
        logger.info(f"✅ Safety mechanism worked - episode was auto-skipped")
    else:
        logger.info(f"✅ Episode completed successfully (environment fix worked)")

    logger.info(f"\n{'='*80}")
    logger.info("Test Complete!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
