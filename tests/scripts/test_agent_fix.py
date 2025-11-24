#!/usr/bin/env python3
"""Quick test to verify DQN Agent memory leak fix"""

import sys
import yaml
from pathlib import Path
import logging
import psutil
import os
import numpy as np
import torch
import gc

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from agents import DQNAgent
import gymnasium as gym


def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


logging.basicConfig(level=logging.WARNING)

config_path = Path(__file__).parent.parent / 'config' / 'diagnostic_config.yaml'
with open(config_path) as f:
    config = yaml.safe_load(f)

obs_space = gym.spaces.Box(-np.inf, np.inf, (10, 12), dtype=np.float32)
act_space = gym.spaces.Discrete(11)

print("="*60)
print("DQN Agent Memory Leak Test (After Fix)")
print("="*60)

mem_start = get_memory_mb()
print(f"Start: {mem_start:.1f} MB")

agent = DQNAgent(obs_space, act_space, config)
mem_after_init = get_memory_mb()
print(f"After init: {mem_after_init:.1f} MB (+{mem_after_init - mem_start:.1f} MB)")

# Train for 10000 updates
for i in range(10000):
    state = np.random.randn(10, 12).astype(np.float32)
    action = agent.select_action(state, deterministic=False)
    reward = np.random.randn()
    next_state = np.random.randn(10, 12).astype(np.float32)
    done = False

    agent.store_experience(state, action, reward, next_state, done)

    if i % 10 == 0:
        loss = agent.update()

    if (i + 1) % 2000 == 0:
        mem_now = get_memory_mb()
        print(f"After {i+1} updates: {mem_now:.1f} MB (+{mem_now - mem_after_init:.1f} MB)")

mem_after_train = get_memory_mb()
print(f"\nAfter 10000 updates: {mem_after_train:.1f} MB (+{mem_after_train - mem_after_init:.1f} MB)")

# Cleanup
del agent
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

mem_after_cleanup = get_memory_mb()
print(f"After cleanup: {mem_after_cleanup:.1f} MB ({mem_after_cleanup - mem_start:+.1f} MB from start)")

leak = mem_after_train - mem_after_init
print(f"\n{'='*60}")
if leak > 100:
    print(f"❌ STILL LEAKING: {leak:.1f} MB over 10000 updates")
    print(f"   Per-update leak: {leak / 10000:.3f} MB")
else:
    print(f"✅ LEAK FIXED: Only {leak:.1f} MB over 10000 updates")
    print(f"   Per-update leak: {leak / 10000:.3f} MB (acceptable)")
print(f"{'='*60}")
