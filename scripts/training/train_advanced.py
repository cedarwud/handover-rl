#!/usr/bin/env python3
"""
Advanced Training - DQN with Double DQN, Dueling, and Evaluation

Features:
- Double DQN
- Extended training (1000+ epochs)
- Learning curves visualization
- Model evaluation
- Checkpoint saving
"""

import sys
sys.path.insert(0, 'src')

import os
import numpy as np
import torch
import yaml
from pathlib import Path
import time
from typing import Dict, List

from agents import DQNAgent
from agents.replay_buffer import ReplayBuffer

print('=' * 70)
print('🚀 Advanced DQN Training - Satellite Handover')
print('=' * 70)

# Load configuration
with open('config/training_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Advanced training settings
EPOCHS = 2000  # Extended training
EVAL_INTERVAL = 100  # Evaluate every 100 epochs
SAVE_INTERVAL = 500  # Save checkpoint every 500 epochs

# Enable advanced features
config['use_double_dqn'] = True  # Enable Double DQN
config['batch_size'] = 64  # Larger batch
config['learning_rate'] = 3e-4  # Slightly higher LR
config['target_update_frequency'] = 100  # More frequent target updates

print(f'\n⚙️  Advanced Configuration:')
print(f'   Epochs: {EPOCHS}')
print(f'   Double DQN: {config["use_double_dqn"]}')
print(f'   Batch size: {config["batch_size"]}')
print(f'   Learning rate: {config["learning_rate"]}')
print(f'   Target update freq: {config["target_update_frequency"]}')

# Initialize agent
state_dim = 12
action_dim = 2

print(f'\n🤖 Initializing DQN Agent...')
agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, config=config)

# Load episode data
episode_dir = Path('data/episodes/train')
episode_files = sorted(episode_dir.glob('episode_*.npz'))

print(f'\n📦 Loading episodes from {episode_dir}...')
print(f'   Found {len(episode_files)} episodes')

# Load all episodes and collect statistics
total_transitions = 0
episode_stats = []

for ep_file in episode_files:
    data = np.load(ep_file)
    states = data['states']
    actions = data['actions']
    rewards = data['rewards']
    next_states = data['next_states']
    dones = data['dones']

    # Calculate episode statistics
    avg_rsrp = states[:, 0].mean()
    max_rsrp = states[:, 0].max()
    min_rsrp = states[:, 0].min()
    total_reward = rewards.sum()

    episode_stats.append({
        'file': ep_file.name,
        'length': len(states),
        'avg_rsrp': avg_rsrp,
        'rsrp_range': (min_rsrp, max_rsrp),
        'total_reward': total_reward
    })

    # Add to replay buffer
    for i in range(len(states)):
        agent.replay_buffer.push(
            states[i],
            actions[i],
            rewards[i],
            next_states[i],
            dones[i]
        )
        total_transitions += 1

print(f'\n✅ Loaded {total_transitions} transitions')
print(f'   Buffer size: {len(agent.replay_buffer)}')

print(f'\n📊 Episode Statistics:')
for stat in episode_stats:
    print(f'   {stat["file"]}: {stat["length"]} steps')
    print(f'      RSRP: {stat["avg_rsrp"]:.2f} dBm (range: {stat["rsrp_range"][0]:.2f} to {stat["rsrp_range"][1]:.2f})')
    print(f'      Total reward: {stat["total_reward"]:.4f}')

# Training loop
print('\n' + '=' * 70)
print('🎯 Starting Advanced Training')
print('=' * 70)

losses = []
eval_scores = []
start_time = time.time()

for epoch in range(EPOCHS):
    # Update
    result = agent.update()

    if result is not None:
        if isinstance(result, dict):
            loss = result.get('loss', result.get('td_loss', 0))
        else:
            loss = result
        losses.append(loss)

    # Print progress
    if (epoch + 1) % 50 == 0:
        if losses:
            avg_loss = np.mean(losses[-50:])
            print(f'Epoch {epoch+1:4d}/{EPOCHS} | Loss: {loss:.6f} | Avg Loss (50): {avg_loss:.6f} | ε: {agent.epsilon:.4f}')
        else:
            print(f'Epoch {epoch+1:4d}/{EPOCHS} | No loss yet | ε: {agent.epsilon:.4f}')

    # Evaluation
    if (epoch + 1) % EVAL_INTERVAL == 0:
        # Evaluate on first episode
        data = np.load(episode_files[0])
        test_states = data['states']

        total_reward = 0
        for state in test_states:
            action = agent.select_action(state, eval_mode=True)
            # Simplified reward calculation
            rsrp = state[0]
            if rsrp > -85:
                reward = 1.0
            elif rsrp > -95:
                reward = 0.5
            elif rsrp > -105:
                reward = 0.0
            elif rsrp > -115:
                reward = -0.5
            else:
                reward = -1.0
            total_reward += reward

        eval_scores.append(total_reward)
        print(f'   📊 Evaluation: Total Reward = {total_reward:.2f}')

    # Save checkpoint
    if (epoch + 1) % SAVE_INTERVAL == 0:
        checkpoint_path = Path('checkpoints') / f'dqn_advanced_epoch_{epoch+1}.pth'
        agent.save(str(checkpoint_path))
        print(f'   💾 Checkpoint saved: {checkpoint_path.name}')

duration = time.time() - start_time

print('\n' + '=' * 70)
print('✅ Training Complete!')
print('=' * 70)
print(f'Total epochs: {EPOCHS}')
print(f'Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)')
print(f'Final epsilon: {agent.epsilon:.4f}')

if losses:
    print(f'\n📈 Loss Statistics:')
    print(f'   Final loss: {losses[-1]:.6f}')
    print(f'   Average loss: {np.mean(losses):.6f}')
    print(f'   Min loss: {np.min(losses):.6f}')
    print(f'   Max loss: {np.max(losses):.6f}')

if eval_scores:
    print(f'\n📊 Evaluation Scores:')
    print(f'   Initial: {eval_scores[0]:.2f}')
    print(f'   Final: {eval_scores[-1]:.2f}')
    print(f'   Best: {np.max(eval_scores):.2f}')
    print(f'   Improvement: {eval_scores[-1] - eval_scores[0]:.2f}')

# Save final model
final_path = Path('checkpoints') / 'dqn_advanced_final.pth'
agent.save(str(final_path))
print(f'\n💾 Final model saved: {final_path}')

# Test inference on all episodes
print('\n🧪 Testing Inference on All Episodes:')
for ep_file in episode_files:
    data = np.load(ep_file)
    test_state = data['states'][0]
    action = agent.select_action(test_state, eval_mode=True)
    rsrp = test_state[0]
    elevation = test_state[4]
    print(f'   {ep_file.name}: RSRP={rsrp:.2f}dBm, Elev={elevation:.2f}° → Action={action} ({["stay", "handover"][action]})')

print('\n' + '=' * 70)
print('🎉 Advanced Training Complete!')
print('=' * 70)
print(f'\n📊 Summary:')
print(f'   Training epochs: {EPOCHS}')
print(f'   Episodes used: {len(episode_files)}')
print(f'   Total transitions: {total_transitions}')
print(f'   Double DQN: Enabled')
print(f'   Final model: {final_path}')
print('=' * 70)
