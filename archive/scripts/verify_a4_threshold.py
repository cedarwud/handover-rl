#!/usr/bin/env python3
"""
Verify A4 Threshold Configuration

Check if A4 RSRP threshold is set to unreasonable values
causing all events to trivially trigger.
"""

import json
import numpy as np
from pathlib import Path

# Load Stage 6 output
orbit_engine_root = Path(__file__).parent.parent / 'orbit-engine'
stage6_dir = orbit_engine_root / 'data' / 'outputs' / 'rl_training' / 'stage6'
stage6_file = sorted(stage6_dir.glob('stage6_research_optimization_*.json'))[-1]

print("=" * 80)
print("A4 Threshold Configuration Analysis")
print("=" * 80)

with open(stage6_file, 'r') as f:
    data = json.load(f)

# Extract RSRP values from A4 events
a4_events = data.get('gpp_events_candidate', {}).get('a4_events', [])

neighbor_rsrp_values = []
serving_rsrp_values = []
trigger_margins = []

for event in a4_events:
    measurements = event.get('measurements', {})
    neighbor_rsrp = measurements.get('neighbor_rsrp_dbm')
    serving_rsrp = measurements.get('serving_rsrp_dbm')
    trigger_margin = measurements.get('trigger_margin_db')

    if neighbor_rsrp is not None:
        neighbor_rsrp_values.append(neighbor_rsrp)
    if serving_rsrp is not None:
        serving_rsrp_values.append(serving_rsrp)
    if trigger_margin is not None:
        trigger_margins.append(trigger_margin)

print(f"\n📊 A4 Events Analysis (N={len(a4_events)}):")

if neighbor_rsrp_values:
    print(f"\nNeighbor RSRP Distribution:")
    print(f"  Min:    {min(neighbor_rsrp_values):>7.2f} dBm")
    print(f"  10th:   {np.percentile(neighbor_rsrp_values, 10):>7.2f} dBm")
    print(f"  Median: {np.percentile(neighbor_rsrp_values, 50):>7.2f} dBm")
    print(f"  90th:   {np.percentile(neighbor_rsrp_values, 90):>7.2f} dBm")
    print(f"  Max:    {max(neighbor_rsrp_values):>7.2f} dBm")
    print(f"  Mean:   {np.mean(neighbor_rsrp_values):>7.2f} dBm")

if serving_rsrp_values:
    print(f"\nServing RSRP Distribution:")
    print(f"  Min:    {min(serving_rsrp_values):>7.2f} dBm")
    print(f"  10th:   {np.percentile(serving_rsrp_values, 10):>7.2f} dBm")
    print(f"  Median: {np.percentile(serving_rsrp_values, 50):>7.2f} dBm")
    print(f"  90th:   {np.percentile(serving_rsrp_values, 90):>7.2f} dBm")
    print(f"  Max:    {max(serving_rsrp_values):>7.2f} dBm")
    print(f"  Mean:   {np.mean(serving_rsrp_values):>7.2f} dBm")

if trigger_margins:
    print(f"\nTrigger Margin Distribution:")
    print(f"  Min:    {min(trigger_margins):>7.2f} dB")
    print(f"  10th:   {np.percentile(trigger_margins, 10):>7.2f} dB")
    print(f"  Median: {np.percentile(trigger_margins, 50):>7.2f} dB")
    print(f"  90th:   {np.percentile(trigger_margins, 90):>7.2f} dB")
    print(f"  Max:    {max(trigger_margins):>7.2f} dB")
    print(f"  Mean:   {np.mean(trigger_margins):>7.2f} dB")

# Load config to check threshold
config_file = orbit_engine_root / 'config' / 'stage6_research_optimization_config.yaml'
import yaml
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

a4_threshold = config['gpp_events']['a4']['rsrp_threshold_dbm']

print(f"\n⚙️  Configured A4 Threshold: {a4_threshold} dBm")

print(f"\n🚨 Problem Analysis:")

if neighbor_rsrp_values:
    min_margin = min(neighbor_rsrp_values) - a4_threshold
    mean_margin = np.mean(neighbor_rsrp_values) - a4_threshold

    print(f"  1. ALL neighbor RSRP values are {min_margin:.1f} to {max(neighbor_rsrp_values) - a4_threshold:.1f} dB above threshold")
    print(f"  2. Mean trigger margin: {mean_margin:.1f} dB (should be 3-10 dB for realistic handover)")
    print(f"  3. Threshold is {min(neighbor_rsrp_values) - a4_threshold:.1f} dB below the WORST case RSRP")

    # Recommend proper threshold
    recommended_threshold = np.percentile(neighbor_rsrp_values, 30)
    print(f"\n💡 Recommendation:")
    print(f"  Set A4 threshold to: {recommended_threshold:.1f} dBm (30th percentile)")
    print(f"  This would create:")
    print(f"    - 30% of neighbors would NOT trigger (maintain decision)")
    print(f"    - 70% of neighbors would trigger (handover decision)")
    print(f"    - Realistic trigger margins: 0-{max(neighbor_rsrp_values) - recommended_threshold:.1f} dB")
    print(f"    - Better learning task for BC model")

print("\n" + "=" * 80)
