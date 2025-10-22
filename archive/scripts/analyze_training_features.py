#!/usr/bin/env python3
"""
Analyze Training Features to Find Remaining Data Leakage

After threshold correction, still getting 100% accuracy.
This script analyzes the feature distributions to find the root cause.
"""

import sys
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from adapters.handover_event_loader import create_handover_event_loader

# Load data
loader = create_handover_event_loader()
orbit_engine_root = Path(__file__).parent.parent / 'orbit-engine'
stage6_dir = orbit_engine_root / 'data' / 'outputs' / 'rl_training' / 'stage6'
a4_events, d2_events = loader.load_latest_events(stage6_dir)

# Load Stage 5
stage5_dir = orbit_engine_root / 'data' / 'outputs' / 'rl_training' / 'stage5'
stage5_file = sorted(stage5_dir.glob('stage5_signal_analysis_*.json'))[-1]

with open(stage5_file, 'r') as f:
    stage5_data = json.load(f)

signal_analysis = stage5_data['signal_analysis']

print("=" * 80)
print("Feature Distribution Analysis - Finding Data Leakage")
print("=" * 80)

# New threshold
NEW_A4_THRESHOLD = -34.5

# Analyze HANDOVER events (after filtering)
handover_features = {
    'neighbor_rsrp': [],
    'serving_rsrp': [],
    'rsrp_diff': [],
    'trigger_margin': [],
    'distance': []
}

for event in a4_events + d2_events:
    measurements = event.get('measurements', {})

    neighbor_rsrp = measurements.get('neighbor_rsrp_dbm')
    serving_rsrp = measurements.get('serving_rsrp_dbm')

    if neighbor_rsrp is None or serving_rsrp is None:
        continue

    # Recalculate trigger margin with new threshold
    trigger_margin = neighbor_rsrp - NEW_A4_THRESHOLD - 2.0

    # Only include if satisfies NEW threshold
    if trigger_margin > 0:
        handover_features['neighbor_rsrp'].append(neighbor_rsrp)
        handover_features['serving_rsrp'].append(serving_rsrp)
        handover_features['rsrp_diff'].append(neighbor_rsrp - serving_rsrp)
        handover_features['trigger_margin'].append(trigger_margin)

        if event.get('event_type') == 'D2':
            distance = measurements.get('neighbor_ground_distance_km', 1000.0)
        else:
            distance = measurements.get('neighbor_distance_km', 1000.0)

        handover_features['distance'].append(distance)

# Analyze MAINTAIN moments (negative examples)
maintain_features = {
    'neighbor_rsrp': [],
    'serving_rsrp': [],
    'rsrp_diff': [],
    'trigger_margin': [],
    'distance': []
}

sample_count = 0
max_samples = 10000

for sat_id, sat_data in signal_analysis.items():
    if sample_count >= max_samples:
        break

    if 'time_series' not in sat_data:
        continue

    time_series = sat_data['time_series']

    for tp in time_series:
        if sample_count >= max_samples:
            break

        signal_quality = tp.get('signal_quality', {})
        rsrp_dbm = signal_quality.get('rsrp_dbm', -100.0)
        distance_km = tp.get('distance_km', 1000.0)

        # Simulate maintain scenario
        # Key: neighbor slightly worse than serving
        neighbor_rsrp = rsrp_dbm + np.random.uniform(-3, 0)
        serving_rsrp = rsrp_dbm
        rsrp_diff = neighbor_rsrp - serving_rsrp
        trigger_margin = neighbor_rsrp - NEW_A4_THRESHOLD - 2.0

        maintain_features['neighbor_rsrp'].append(neighbor_rsrp)
        maintain_features['serving_rsrp'].append(serving_rsrp)
        maintain_features['rsrp_diff'].append(rsrp_diff)
        maintain_features['trigger_margin'].append(trigger_margin)
        maintain_features['distance'].append(distance_km)

        sample_count += 1

print(f"\n📊 Sample Sizes:")
print(f"  Handover: {len(handover_features['rsrp_diff'])}")
print(f"  Maintain: {len(maintain_features['rsrp_diff'])}")

print(f"\n🔍 Feature Analysis:")

# RSRP Diff
print(f"\nRSRP Difference (Neighbor - Serving):")
print(f"  Handover: min={min(handover_features['rsrp_diff']):.2f}, max={max(handover_features['rsrp_diff']):.2f}, mean={np.mean(handover_features['rsrp_diff']):.2f} dB")
print(f"  Maintain: min={min(maintain_features['rsrp_diff']):.2f}, max={max(maintain_features['rsrp_diff']):.2f}, mean={np.mean(maintain_features['rsrp_diff']):.2f} dB")

# Check overlap
handover_min_diff = min(handover_features['rsrp_diff'])
maintain_max_diff = max(maintain_features['rsrp_diff'])

if handover_min_diff > maintain_max_diff:
    print(f"  ❌ NO OVERLAP! Handover min ({handover_min_diff:.2f}) > Maintain max ({maintain_max_diff:.2f})")
    print(f"  🚨 DATA LEAKAGE: Model can use simple rule: if rsrp_diff > {maintain_max_diff:.2f}: handover")
else:
    print(f"  ✅ Overlap exists: ranges overlap from {handover_min_diff:.2f} to {maintain_max_diff:.2f} dB")

# Trigger Margin
print(f"\nTrigger Margin (with new threshold -34.5 dBm):")
print(f"  Handover: min={min(handover_features['trigger_margin']):.2f}, max={max(handover_features['trigger_margin']):.2f}, mean={np.mean(handover_features['trigger_margin']):.2f} dB")
print(f"  Maintain: min={min(maintain_features['trigger_margin']):.2f}, max={max(maintain_features['trigger_margin']):.2f}, mean={np.mean(maintain_features['trigger_margin']):.2f} dB")

# Check negative margins
handover_negative = sum(1 for m in handover_features['trigger_margin'] if m < 0)
maintain_negative = sum(1 for m in maintain_features['trigger_margin'] if m < 0)

print(f"  Handover negative margins: {handover_negative}/{len(handover_features['trigger_margin'])} ({handover_negative/len(handover_features['trigger_margin'])*100:.1f}%)")
print(f"  Maintain negative margins: {maintain_negative}/{len(maintain_features['trigger_margin'])} ({maintain_negative/len(maintain_features['trigger_margin'])*100:.1f}%)")

# Distance
print(f"\nDistance:")
print(f"  Handover: mean={np.mean(handover_features['distance']):.2f} km")
print(f"  Maintain: mean={np.mean(maintain_features['distance']):.2f} km")

print(f"\n🚨 Root Cause Analysis:")
print(f"\n1. RSRP Difference Problem:")
print(f"   - Handover events: neighbor is BETTER than serving ({np.mean(handover_features['rsrp_diff']):.2f} dB better)")
print(f"   - Maintain examples: neighbor is WORSE than serving ({np.mean(maintain_features['rsrp_diff']):.2f} dB worse)")
print(f"   - This is because handover events are ONLY triggered when neighbor is better!")
print(f"   - Solution: Need maintain examples where neighbor is also better, but NOT enough to trigger")

print(f"\n2. Our Negative Sampling Strategy is WRONG:")
print(f"   - We sample random time points and make neighbor WORSE (rsrp - 0 to 3 dB)")
print(f"   - But real maintain scenarios happen when neighbor is BETTER, just not enough")
print(f"   - Example: neighbor is +2 dB better, but threshold requires +3 dB minimum")

print(f"\n💡 Correct Strategy:")
print(f"   1. For A4 events: Use Stage 4 candidate pool data")
print(f"   2. Extract ALL visible neighbors at each time point")
print(f"   3. Label as 'handover' if neighbor_rsrp > threshold + margin")
print(f"   4. Label as 'maintain' if neighbor_rsrp < threshold + margin")
print(f"   5. This creates realistic boundary: some neighbors close to threshold")

print("=" * 80)
