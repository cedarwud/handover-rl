#!/usr/bin/env python3
"""
Pinpoint Memory Leak Source

Strategy: Test each component separately to identify leak source
"""

import sys
import yaml
import gc
from pathlib import Path
from datetime import datetime, timedelta
import logging
import psutil
import os
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from adapters import AdapterWrapper
from environments.satellite_handover_env import SatelliteHandoverEnv
from agents import DQNAgent
from trainers import OffPolicyTrainer
from utils.satellite_utils import load_stage4_optimized_satellites
from configs import get_level_config


def get_memory_mb():
    """Get current RSS memory in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def setup_logging():
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise
        format='%(message)s',
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def test_replay_buffer_leak():
    """Test if replay buffer leaks memory"""
    from agents.replay_buffer import ReplayBuffer

    print("\n" + "="*60)
    print("Test 1: Replay Buffer")
    print("="*60)

    mem_start = get_memory_mb()
    buffer = ReplayBuffer(capacity=10000)

    # Fill buffer with data
    for i in range(15000):  # More than capacity to test rotation
        state = np.random.randn(10, 12).astype(np.float32)
        action = np.random.randint(0, 11)
        reward = np.random.randn()
        next_state = np.random.randn(10, 12).astype(np.float32)
        done = False
        buffer.push(state, action, reward, next_state, done)

    mem_after_fill = get_memory_mb()
    print(f"After filling buffer (15000 items): +{mem_after_fill - mem_start:.1f} MB")

    # Sample many times
    for _ in range(1000):
        if buffer.is_ready(64):
            batch = buffer.sample(64)

    mem_after_sample = get_memory_mb()
    print(f"After 1000 samples: +{mem_after_sample - mem_after_fill:.1f} MB")

    # Clear buffer
    buffer.clear()
    gc.collect()

    mem_after_clear = get_memory_mb()
    print(f"After clear + gc: {mem_after_clear - mem_start:+.1f} MB from start")

    if abs(mem_after_clear - mem_start) < 10:
        print("✅ Replay Buffer: NO LEAK")
    else:
        print(f"❌ Replay Buffer: LEAKS ~{mem_after_clear - mem_start:.1f} MB")


def test_agent_leak():
    """Test if DQN agent leaks memory"""
    import torch

    print("\n" + "="*60)
    print("Test 2: DQN Agent")
    print("="*60)

    config_path = Path(__file__).parent.parent / 'config' / 'diagnostic_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    import gymnasium as gym
    obs_space = gym.spaces.Box(-np.inf, np.inf, (10, 12), dtype=np.float32)
    act_space = gym.spaces.Discrete(11)

    mem_start = get_memory_mb()

    agent = DQNAgent(obs_space, act_space, config)
    mem_after_init = get_memory_mb()
    print(f"After agent init: +{mem_after_init - mem_start:.1f} MB")

    # Train for many steps
    for i in range(10000):
        state = np.random.randn(10, 12).astype(np.float32)
        action = agent.select_action(state, deterministic=False)

        reward = np.random.randn()
        next_state = np.random.randn(10, 12).astype(np.float32)
        done = False

        agent.store_experience(state, action, reward, next_state, done)

        if i % 10 == 0:
            loss = agent.update()

    mem_after_train = get_memory_mb()
    print(f"After 10000 updates: +{mem_after_train - mem_after_init:.1f} MB")

    # Clean up
    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    mem_after_cleanup = get_memory_mb()
    print(f"After cleanup: {mem_after_cleanup - mem_start:+.1f} MB from start")

    if abs(mem_after_cleanup - mem_start) < 50:
        print("✅ DQN Agent: NO SIGNIFICANT LEAK")
    else:
        print(f"❌ DQN Agent: LEAKS ~{mem_after_cleanup - mem_start:.1f} MB")


def test_environment_leak():
    """Test if environment leaks memory"""
    print("\n" + "="*60)
    print("Test 3: SatelliteHandoverEnv")
    print("="*60)

    config_path = Path(__file__).parent.parent / 'config' / 'diagnostic_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mem_start = get_memory_mb()

    adapter = AdapterWrapper(config)
    mem_after_adapter = get_memory_mb()
    print(f"After adapter init: +{mem_after_adapter - mem_start:.1f} MB")

    satellite_ids = load_stage4_optimized_satellites()
    env = SatelliteHandoverEnv(adapter, satellite_ids, config)
    mem_after_env = get_memory_mb()
    print(f"After env init: +{mem_after_env - mem_after_adapter:.1f} MB")

    # Run 50 episodes
    start_time_base = datetime(2025, 10, 10, 0, 0, 0)
    episode_stride_minutes = 10

    for ep in range(50):
        time_offset = ep * episode_stride_minutes
        ep_start = start_time_base + timedelta(minutes=time_offset)

        obs, info = env.reset(options={'start_time': ep_start})

        for step in range(100):  # Only 100 steps per episode for speed
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        if (ep + 1) % 10 == 0:
            mem_now = get_memory_mb()
            print(f"After {ep+1} episodes: {mem_now - mem_after_env:+.1f} MB")

    mem_after_episodes = get_memory_mb()
    print(f"After 50 episodes: +{mem_after_episodes - mem_after_env:.1f} MB")

    # Cleanup
    del env
    del adapter
    gc.collect()

    mem_after_cleanup = get_memory_mb()
    print(f"After cleanup: {mem_after_cleanup - mem_start:+.1f} MB from start")

    leak_per_episode = (mem_after_episodes - mem_after_env) / 50
    print(f"\nLeak per episode: ~{leak_per_episode:.2f} MB")

    if leak_per_episode > 10:
        print(f"❌ Environment: SIGNIFICANT LEAK ({leak_per_episode:.1f} MB/episode)")
    elif leak_per_episode > 1:
        print(f"⚠️  Environment: MINOR LEAK ({leak_per_episode:.1f} MB/episode)")
    else:
        print(f"✅ Environment: NO SIGNIFICANT LEAK")


def test_hdf5_leak():
    """Test if HDF5 precompute table leaks memory"""
    print("\n" + "="*60)
    print("Test 4: HDF5 PrecomputeTable")
    print("="*60)

    config_path = Path(__file__).parent.parent / 'config' / 'diagnostic_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mem_start = get_memory_mb()

    adapter = AdapterWrapper(config)
    mem_after_init = get_memory_mb()
    print(f"After adapter/HDF5 init: +{mem_after_init - mem_start:.1f} MB")

    # Access data many times
    test_time = datetime(2025, 10, 11, 12, 0, 0)
    sat_id = '45047'

    for i in range(10000):
        data = adapter.get_state(sat_id, test_time)

        if (i + 1) % 2000 == 0:
            mem_now = get_memory_mb()
            print(f"After {i+1} accesses: {mem_now - mem_after_init:+.1f} MB")

    mem_after_access = get_memory_mb()
    print(f"After 10000 accesses: +{mem_after_access - mem_after_init:.1f} MB")

    del adapter
    gc.collect()

    mem_after_cleanup = get_memory_mb()
    print(f"After cleanup: {mem_after_cleanup - mem_start:+.1f} MB from start")

    if abs(mem_after_access - mem_after_init) > 100:
        print(f"❌ HDF5: SIGNIFICANT LEAK ({mem_after_access - mem_after_init:.1f} MB)")
    else:
        print(f"✅ HDF5: NO SIGNIFICANT LEAK")


def main():
    print("="*60)
    print("Memory Leak Source Identification")
    print("="*60)

    # Test each component
    test_replay_buffer_leak()
    test_agent_leak()
    test_hdf5_leak()
    test_environment_leak()

    print("\n" + "="*60)
    print("Diagnosis Complete")
    print("="*60)


if __name__ == '__main__':
    main()
