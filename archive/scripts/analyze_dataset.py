#!/usr/bin/env python3
"""
Analyze Dataset to Find Why 100% Accuracy

Check feature distributions to identify potential data leakage
"""

import sys
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
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
print("Dataset Analysis - Finding Data Leakage")
print("=" * 80)

# Analyze handover events
handover_rsrp_diffs = []
handover_trigger_margins = []

for event in a4_events + d2_events:
    measurements = event['measurements']
    neighbor_rsrp = measurements.get('neighbor_rsrp_dbm', -100)
    serving_rsrp = measurements.get('serving_rsrp_dbm', measurements.get('threshold_dbm', -100))
    rsrp_diff = neighbor_rsrp - serving_rsrp
    trigger_margin = measurements.get('trigger_margin_db', measurements.get('trigger_margin_km', 0))

    handover_rsrp_diffs.append(rsrp_diff)
    handover_trigger_margins.append(trigger_margin)

# Analyze non-handover moments
maintain_rsrp_diffs = []
sample_count = 0
max_samples = 10000

for sat_id, sat_data in signal_analysis.items():
    if 'time_series' not in sat_data:
        continue

    for tp in sat_data['time_series']:
        if sample_count >= max_samples:
            break

        signal_quality = tp.get('signal_quality', {})
        rsrp = signal_quality.get('rsrp_dbm', -100)

        # For maintain: no neighbor, so diff is 0 or small
        # This is the problem! We don't have neighbor info in maintain moments
        maintain_rsrp_diffs.append(0.0)  # This is artificial!
        sample_count += 1

print(f"\n📊 Feature Analysis:")
print(f"\nHandover Events (N={len(handover_rsrp_diffs)}):")
print(f"  RSRP Diff: min={min(handover_rsrp_diffs):.2f}, max={max(handover_rsrp_diffs):.2f}, mean={np.mean(handover_rsrp_diffs):.2f}")
print(f"  Trigger Margin: min={min(handover_trigger_margins):.2f}, max={max(handover_trigger_margins):.2f}, mean={np.mean(handover_trigger_margins):.2f}")

print(f"\nMaintain Moments (N={len(maintain_rsrp_diffs)}):")
print(f"  RSRP Diff: min={min(maintain_rsrp_diffs):.2f}, max={max(maintain_rsrp_diffs):.2f}, mean={np.mean(maintain_rsrp_diffs):.2f}")

print(f"\n🚨 PROBLEM IDENTIFIED:")
print(f"  1. Handover RSRP diff: {np.mean(handover_rsrp_diffs):.2f} dB (neighbor better)")
print(f"  2. Maintain RSRP diff: {np.mean(maintain_rsrp_diffs):.2f} dB (no difference)")
print(f"  3. Trigger margin only exists in handover events!")
print(f"\n  => Model can easily distinguish by RSRP diff or trigger margin")
print(f"  => This is DATA LEAKAGE - features directly reveal the label")

print(f"\n💡 Solution:")
print(f"  We need REAL neighbor satellite data for maintain moments")
print(f"  This requires:")
print(f"    1. Extract all visible satellites at each time point from Stage 4")
print(f"    2. Compare serving vs best neighbor (even if no handover)")
print(f"    3. Label: handover if diff > threshold, else maintain")

print("=" * 80)
