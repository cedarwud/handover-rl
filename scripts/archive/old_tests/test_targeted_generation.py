#!/usr/bin/env python3
"""
Targeted Episode Generation - Generate episodes during visible satellite passes
"""

import sys
sys.path.insert(0, 'src')

from datetime import datetime, timedelta
import yaml
import time
import numpy as np
from pathlib import Path

from adapters import OrbitEngineAdapter

print('=' * 70)
print('🎯 Targeted Episode Generation - Visible Passes Only')
print('=' * 70)

# Load configuration
with open('config/data_gen_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize adapter
adapter = OrbitEngineAdapter(config)

satellite_id = config["data_generation"]["satellite_ids"][0]  # STARLINK-1008
print(f'🛰️  Target satellite: {satellite_id}')
print(f'   Ground Station: (24.9441°N, 121.3714°E)\n')

# Find visible pass around 08:40 (max elevation)
episode_start = datetime(2025, 10, 16, 8, 0, 0)  # Start at 08:00
episode_duration_minutes = 95  # Starlink orbital period
time_step_seconds = 5

print(f'📅 Episode window: {episode_start.strftime("%Y-%m-%d %H:%M")} + {episode_duration_minutes}min')
print(f'   Time step: {time_step_seconds}s')
print(f'   Total steps: {episode_duration_minutes * 60 // time_step_seconds}')
print('\n' + '=' * 70)

# Generate timestamps
timestamps = [
    episode_start + timedelta(seconds=time_step_seconds * i)
    for i in range(episode_duration_minutes * 60 // time_step_seconds)
]

# Calculate states
print(f'\n📊 Calculating states for {len(timestamps)} timesteps...')
states = []
valid_count = 0
connectable_count = 0

for i, timestamp in enumerate(timestamps):
    try:
        state = adapter.calculate_state(
            satellite_id=satellite_id,
            timestamp=timestamp
        )

        if state:
            valid_count += 1
            is_connectable = state.get('is_connectable', False)

            if is_connectable:
                connectable_count += 1
                elevation = state.get('elevation_deg', 0)
                rsrp = state.get('rsrp_dbm', -999)

                # Show first few connectable states
                if connectable_count <= 5:
                    print(f'✅ Step {i:4d} ({timestamp.strftime("%H:%M:%S")}) - '
                          f'Elevation: {elevation:6.2f}° | RSRP: {rsrp:7.2f} dBm')

                # Convert state dict to array (12-dim)
                state_array = np.array([
                    state.get('rsrp_dbm', 0),
                    state.get('rsrq_db', 0),
                    state.get('sinr_db', 0),
                    state.get('distance_km', 0),
                    state.get('elevation_deg', 0),
                    state.get('azimuth_deg', 0),
                    state.get('doppler_shift_hz', 0),
                    state.get('path_loss_db', 0),
                    state.get('atmospheric_loss_db', 0),
                    state.get('offset_mo_db', 0),
                    state.get('cell_offset_db', 0),
                    state.get('velocity_km_s', 0)
                ], dtype=np.float32)

                states.append(state_array)

    except Exception as e:
        if valid_count == 0:  # Only show first error
            print(f'❌ Step {i}: {e}')

if connectable_count >= 5:
    print(f'   ... ({connectable_count - 5} more connectable states)')

print('\n' + '=' * 70)
print('📊 Results')
print('=' * 70)
print(f'Total timesteps: {len(timestamps)}')
print(f'Valid states: {valid_count} ({valid_count/len(timestamps):.1%})')
print(f'Connectable states: {connectable_count} ({connectable_count/len(timestamps):.1%})')
print(f'Episode validity: {"✅ PASS" if connectable_count >= len(timestamps) * 0.5 else "❌ FAIL"} '
      f'(threshold: ≥50%)')

if connectable_count > 0:
    states_array = np.array(states)
    print(f'\n📦 Episode data shape: {states_array.shape}')
    print(f'   RSRP range: {states_array[:, 0].min():.2f} to {states_array[:, 0].max():.2f} dBm')
    print(f'   Elevation range: {states_array[:, 4].min():.2f}° to {states_array[:, 4].max():.2f}°')

    # Save as test episode
    output_dir = Path('data/episodes/test')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'test_episode_001.npz'

    # Generate dummy actions and rewards for now
    actions = np.zeros(len(states), dtype=np.int32)
    rewards = np.zeros(len(states), dtype=np.float32)
    next_states = np.roll(states_array, -1, axis=0)
    next_states[-1] = states_array[-1]
    dones = np.zeros(len(states), dtype=np.float32)
    dones[-1] = 1.0

    np.savez_compressed(
        output_file,
        states=states_array,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        timestamps=np.array([t.timestamp() for t in timestamps[:len(states)]])
    )

    print(f'\n✅ Test episode saved: {output_file}')
    print(f'   File size: {output_file.stat().st_size / 1024:.1f} KB')

print('\n' + '=' * 70)
