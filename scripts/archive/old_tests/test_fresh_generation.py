#!/usr/bin/env python3
"""
Fresh Test - Data Generation with All Fixes Applied
"""

import sys
sys.path.insert(0, 'src')

from datetime import datetime
import yaml
import time

from data_generation.rl_data_generator import RLDataGenerator

print('=' * 70)
print('🧪 Fresh Test: Data Generation with All Fixes')
print('=' * 70)

# Load configuration
with open('config/data_gen_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f'✅ Configuration:')
print(f'   Satellites: {config["data_generation"]["satellite_ids"]}')

# Initialize generator
print('\n🔧 Initializing RLDataGenerator...')
generator = RLDataGenerator(config)

# Generate training data
print('\n📦 Generating training data (2025-10-16, 1 day)...')
print('-' * 70)

start_time = time.time()

train_episodes = generator.generate_dataset(
    start_date=datetime(2025, 10, 16, 0, 0, 0),
    end_date=datetime(2025, 10, 17, 0, 0, 0),
    output_dir='data/episodes/train',
    max_episodes=20
)

duration = time.time() - start_time

print('\n' + '=' * 70)
print('✅ Data generation complete!')
print('=' * 70)
print(f'Generated episodes: {train_episodes}')
print(f'Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)')
print('=' * 70)
