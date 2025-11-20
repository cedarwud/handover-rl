#!/usr/bin/env python3
"""
Memory Leak Diagnostic Tool

Tests 100 episodes and tracks memory usage in detail to identify leak source.

Strategy:
1. Track memory before/after each episode
2. Track memory before/after each component (agent update, environment step)
3. Profile specific functions with memory_profiler
4. Generate memory growth curve
"""

import sys
import yaml
import gc
from pathlib import Path
from datetime import datetime, timedelta
import logging
import psutil
import os
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from adapters import AdapterWrapper
from environments.satellite_handover_env import SatelliteHandoverEnv
from agents import DQNAgent
from trainers import OffPolicyTrainer
from utils.satellite_utils import load_stage4_optimized_satellites
from configs import get_level_config


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent(),
    }


def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    return logging.getLogger(__name__)


def main():
    logger = setup_logging()

    # Start tracemalloc for detailed memory tracking
    tracemalloc.start()

    logger.info("="*80)
    logger.info("Memory Leak Diagnostic - 100 Episodes")
    logger.info("="*80)

    # Load config
    config_path = Path(__file__).parent.parent / 'config' / 'diagnostic_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get level 4 config
    level_config = get_level_config(4)

    # Disable safety checks for this test (we want to see the actual leak)
    if 'training' not in config:
        config['training'] = {}
    config['training'].update({
        'num_episodes': 100,  # Only test 100 episodes
        'enable_safety_checks': False,  # Disable to see actual memory growth
    })

    # Initialize components
    logger.info("Initializing components...")
    mem_start = get_memory_usage()
    logger.info(f"Initial memory: {mem_start['rss_mb']:.1f} MB RSS, {mem_start['vms_mb']:.1f} MB VMS")

    adapter = AdapterWrapper(config)
    satellite_ids = load_stage4_optimized_satellites()
    env = SatelliteHandoverEnv(adapter, satellite_ids, config)
    agent = DQNAgent(env.observation_space, env.action_space, config)
    trainer = OffPolicyTrainer(env, agent, config)

    mem_after_init = get_memory_usage()
    logger.info(f"After init: {mem_after_init['rss_mb']:.1f} MB RSS (+{mem_after_init['rss_mb']-mem_start['rss_mb']:.1f} MB)")

    # Memory tracking arrays
    episode_numbers = []
    memory_rss = []
    memory_vms = []
    memory_percent = []

    # Component-level memory tracking
    agent_update_memory = []
    env_step_memory = []
    gc_collect_memory = []

    # Episode timing
    start_time_base = datetime(2025, 10, 10, 0, 0, 0)
    episode_stride_minutes = int(20 * 0.5)

    logger.info("\n" + "="*80)
    logger.info("Starting Episode Loop - Tracking Memory")
    logger.info("="*80)

    for episode_idx in range(100):
        mem_before_episode = get_memory_usage()

        # Calculate episode start time
        time_offset_minutes = episode_idx * episode_stride_minutes
        episode_start_time = start_time_base + timedelta(minutes=time_offset_minutes)

        # Reset environment
        obs, info = env.reset(options={'start_time': episode_start_time})

        episode_reward = 0.0
        episode_steps = 0
        done = False

        agent.on_episode_start()

        # Episode loop
        while not done:
            # Select action
            action_mask = info.get('action_mask', None)
            action = agent.select_action(obs, deterministic=False, action_mask=action_mask)

            # Environment step
            mem_before_step = get_memory_usage()
            next_obs, reward, terminated, truncated, info = env.step(action)
            mem_after_step = get_memory_usage()
            env_step_memory.append(mem_after_step['rss_mb'] - mem_before_step['rss_mb'])

            done = terminated or truncated

            # Store experience
            if hasattr(agent, 'store_experience'):
                agent.store_experience(obs, action, reward, next_obs, done)

            # Agent update
            if trainer.total_steps % trainer.update_frequency == 0:
                mem_before_update = get_memory_usage()
                loss = agent.update()
                mem_after_update = get_memory_usage()
                agent_update_memory.append(mem_after_update['rss_mb'] - mem_before_update['rss_mb'])

            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            trainer.total_steps += 1

        # Episode end
        stats = info.get('episode_stats', {})
        episode_info = {
            'num_handovers': stats.get('num_handovers', 0),
            'avg_rsrp': stats.get('avg_rsrp', 0.0),
            'num_ping_pongs': stats.get('num_ping_pongs', 0),
        }
        agent.on_episode_end(episode_reward, episode_info)

        # Test gc.collect()
        mem_before_gc = get_memory_usage()
        collected = gc.collect()
        mem_after_gc = get_memory_usage()
        gc_collect_memory.append(mem_after_gc['rss_mb'] - mem_before_gc['rss_mb'])

        # Record memory after episode
        mem_after_episode = get_memory_usage()

        episode_numbers.append(episode_idx)
        memory_rss.append(mem_after_episode['rss_mb'])
        memory_vms.append(mem_after_episode['vms_mb'])
        memory_percent.append(mem_after_episode['percent'])

        # Log every 10 episodes
        if (episode_idx + 1) % 10 == 0:
            mem_growth = mem_after_episode['rss_mb'] - mem_before_episode['rss_mb']
            logger.info(
                f"Episode {episode_idx+1:3d}/100: "
                f"RSS={mem_after_episode['rss_mb']:6.1f} MB (+{mem_growth:5.1f} MB), "
                f"VMS={mem_after_episode['vms_mb']:6.1f} MB, "
                f"Percent={mem_after_episode['percent']:4.1f}%, "
                f"GC freed={-gc_collect_memory[-1]:5.1f} MB, "
                f"Collected={collected} objects"
            )

    # Get tracemalloc snapshot
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    logger.info("\n" + "="*80)
    logger.info("Memory Analysis Results")
    logger.info("="*80)

    # Overall statistics
    mem_final = get_memory_usage()
    total_growth = mem_final['rss_mb'] - mem_after_init['rss_mb']
    growth_per_episode = total_growth / 100

    logger.info(f"\n📊 Overall Memory Growth:")
    logger.info(f"  Initial (after init): {mem_after_init['rss_mb']:.1f} MB")
    logger.info(f"  Final (after 100 ep): {mem_final['rss_mb']:.1f} MB")
    logger.info(f"  Total growth: {total_growth:.1f} MB")
    logger.info(f"  Growth per episode: {growth_per_episode:.2f} MB")
    logger.info(f"  Estimated for 1000 ep: {growth_per_episode * 1000:.1f} MB ({growth_per_episode * 1000 / 1024:.1f} GB)")

    # Component analysis
    logger.info(f"\n🔍 Component-Level Analysis:")
    logger.info(f"  Agent update avg growth: {np.mean(agent_update_memory):.3f} MB")
    logger.info(f"  Env step avg growth: {np.mean(env_step_memory):.3f} MB")
    logger.info(f"  GC collect avg freed: {np.mean(gc_collect_memory):.3f} MB")
    logger.info(f"  GC effectiveness: {'POOR' if abs(np.mean(gc_collect_memory)) < 0.1 else 'GOOD'}")

    # Top memory allocations (tracemalloc)
    logger.info(f"\n🔝 Top 10 Memory Allocations (by tracemalloc):")
    for idx, stat in enumerate(top_stats[:10], 1):
        logger.info(f"  {idx}. {stat.filename}:{stat.lineno}: {stat.size / 1024 / 1024:.1f} MB")

    # Generate plots
    logger.info(f"\n📈 Generating memory growth plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: RSS Memory Growth
    axes[0, 0].plot(episode_numbers, memory_rss, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('RSS Memory (MB)')
    axes[0, 0].set_title('RSS Memory Growth Over 100 Episodes')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: VMS Memory Growth
    axes[0, 1].plot(episode_numbers, memory_vms, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('VMS Memory (MB)')
    axes[0, 1].set_title('Virtual Memory Growth Over 100 Episodes')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Memory Percent
    axes[1, 0].plot(episode_numbers, memory_percent, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Memory Usage (%)')
    axes[1, 0].set_title('Memory Usage Percentage')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Component Memory Changes
    if len(agent_update_memory) > 0:
        axes[1, 1].hist(agent_update_memory, bins=50, alpha=0.5, label='Agent Update', color='blue')
    if len(env_step_memory) > 0:
        axes[1, 1].hist(env_step_memory, bins=50, alpha=0.5, label='Env Step', color='green')
    if len(gc_collect_memory) > 0:
        axes[1, 1].hist(gc_collect_memory, bins=50, alpha=0.5, label='GC Collect', color='red')
    axes[1, 1].set_xlabel('Memory Change (MB)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Memory Changes by Component')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(__file__).parent.parent / 'logs' / 'memory_leak_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"✅ Plot saved to: {plot_path}")

    # Save detailed data
    data_path = Path(__file__).parent.parent / 'logs' / 'memory_leak_data.npz'
    np.savez(
        data_path,
        episode_numbers=episode_numbers,
        memory_rss=memory_rss,
        memory_vms=memory_vms,
        memory_percent=memory_percent,
        agent_update_memory=agent_update_memory,
        env_step_memory=env_step_memory,
        gc_collect_memory=gc_collect_memory,
    )
    logger.info(f"✅ Data saved to: {data_path}")

    # Conclusion
    logger.info(f"\n" + "="*80)
    logger.info("Diagnosis Complete")
    logger.info("="*80)

    if growth_per_episode > 1.0:
        logger.warning(f"⚠️  MEMORY LEAK DETECTED: {growth_per_episode:.2f} MB per episode")
        logger.warning(f"   At this rate, 1000 episodes will use {growth_per_episode * 1000 / 1024:.1f} GB")
    else:
        logger.info(f"✅ Memory growth is acceptable: {growth_per_episode:.2f} MB per episode")

    if abs(np.mean(gc_collect_memory)) < 0.1:
        logger.warning(f"⚠️  gc.collect() is NOT effective (avg freed: {np.mean(gc_collect_memory):.3f} MB)")
        logger.warning(f"   This suggests the leak is NOT from Python object references")
    else:
        logger.info(f"✅ gc.collect() is effective (avg freed: {np.mean(gc_collect_memory):.3f} MB)")

    tracemalloc.stop()


if __name__ == '__main__':
    main()
