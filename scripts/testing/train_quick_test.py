#!/usr/bin/env python3
"""
Quick Training Test - Train DQN on 4 episodes

Tests the complete training pipeline end-to-end.
"""

import sys
sys.path.insert(0, 'src')

import os
import numpy as np
import torch
import yaml
from pathlib import Path
import time

from agents import DQNAgent
from agents.replay_buffer import ReplayBuffer

print('=' * 70)
print('🚀 Quick Training Test - DQN on 4 Episodes')
print('=' * 70)

# Load configuration
with open('config/training_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Override for quick test
config['num_epochs'] = 50  # Quick test: 50 epochs instead of full training
config['batch_size'] = 32  # Smaller batch for limited data

print(f'✅ Configuration loaded:')
print(f'   Epochs: {config["num_epochs"]}')
print(f'   Batch size: {config["batch_size"]}')
print(f'   Learning rate: {config.get("learning_rate", 1e-4)}')

# Initialize agent
state_dim = 12  # RSRP, RSRQ, SINR, distance, elevation, azimuth, doppler, path_loss, atm_loss, offset_mo, cell_offset, velocity
action_dim = 2  # 0: stay, 1: handover

print(f'\n🤖 Initializing DQN Agent...')
print(f'   State dimension: {state_dim}')
print(f'   Action dimension: {action_dim}')

agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, config=config)

# Load episode data
episode_dir = Path('data/episodes/train')
episode_files = sorted(episode_dir.glob('episode_*.npz'))

print(f'\n📦 Loading episodes from {episode_dir}...')
print(f'   Found {len(episode_files)} episodes')

# Load all episodes into replay buffer
total_transitions = 0
for ep_file in episode_files:
    data = np.load(ep_file)
    states = data['states']
    actions = data['actions']
    rewards = data['rewards']
    next_states = data['next_states']
    dones = data['dones']

    print(f'\n   {ep_file.name}:')
    print(f'      Transitions: {len(states)}')
    print(f'      State shape: {states.shape}')

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

print(f'\n✅ Loaded {total_transitions} transitions into replay buffer')
print(f'   Buffer size: {len(agent.replay_buffer)}')

# Training loop
print('\n' + '=' * 70)
print('🎯 Starting Training')
print('=' * 70)

num_epochs = config['num_epochs']
batch_size = config['batch_size']

losses = []
start_time = time.time()

for epoch in range(num_epochs):
    # Sample batch and update
    result = agent.update()

    if result is not None:
        # Extract loss value (may be dict or scalar)
        if isinstance(result, dict):
            loss = result.get('loss', result.get('td_loss', 0))
        else:
            loss = result

        losses.append(loss)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = np.mean(losses[-10:])
            print(f'Epoch {epoch+1:3d}/{num_epochs} | Loss: {loss:.4f} | Avg Loss (last 10): {avg_loss:.4f} | ε: {agent.epsilon:.3f}')

duration = time.time() - start_time

print('\n' + '=' * 70)
print('✅ Training Complete!')
print('=' * 70)
print(f'Total epochs: {num_epochs}')
print(f'Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)')
print(f'Final epsilon: {agent.epsilon:.3f}')

if losses:
    print(f'Final loss: {losses[-1]:.4f}')
    print(f'Average loss: {np.mean(losses):.4f}')
    print(f'Min loss: {np.min(losses):.4f}')
    print(f'Max loss: {np.max(losses):.4f}')

# Save model
output_dir = Path('checkpoints')
output_dir.mkdir(exist_ok=True)
checkpoint_path = output_dir / 'dqn_quick_test.pth'

print(f'\n💾 Saving model to {checkpoint_path}...')
agent.save(str(checkpoint_path))
print(f'✅ Model saved successfully')

# Test inference
print('\n🧪 Testing inference on first episode state...')
data = np.load(episode_files[0])
test_state = data['states'][0]
print(f'   Test state shape: {test_state.shape}')
print(f'   RSRP: {test_state[0]:.2f} dBm, Elevation: {test_state[4]:.2f}°')

action = agent.select_action(test_state, eval_mode=True)
print(f'   Selected action: {action} ({["stay", "handover"][action]})')

print('\n' + '=' * 70)
print('🎉 Quick Training Test Complete!')
print('=' * 70)
