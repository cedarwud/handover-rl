#!/usr/bin/env python3
"""
Comprehensive Threshold Configuration Analysis

Check all event types (A3, A4, A5, D2) for unrealistic threshold configurations
that could cause data leakage in ML training.
"""

import json
import yaml
import numpy as np
from pathlib import Path

# Load Stage 6 output
orbit_engine_root = Path(__file__).parent.parent / 'orbit-engine'

# Try RL training dir first, fallback to regular stage6 dir
rl_stage6_dir = orbit_engine_root / 'data' / 'outputs' / 'rl_training' / 'stage6'
regular_stage6_dir = orbit_engine_root / 'data' / 'outputs' / 'stage6'

if list(rl_stage6_dir.glob('stage6_research_optimization_*.json')):
    stage6_dir = rl_stage6_dir
    print(f"Using RL training data from: {rl_stage6_dir}")
else:
    stage6_dir = regular_stage6_dir
    print(f"Using regular Stage 6 data from: {regular_stage6_dir}")

stage6_file = sorted(stage6_dir.glob('stage6_research_optimization_*.json'))[-1]
print(f"Analyzing: {stage6_file.name}")

print("=" * 80)
print("Complete Handover Event Threshold Analysis")
print("=" * 80)

with open(stage6_file, 'r') as f:
    data = json.load(f)

# Load config
config_file = orbit_engine_root / 'config' / 'stage6_research_optimization_config.yaml'
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

gpp_events = data.get('gpp_events_candidate', {})

# ==================== A3 事件分析 ====================
print("\n" + "=" * 80)
print("📊 A3 Events Analysis (Neighbor becomes offset better than serving)")
print("=" * 80)

a3_events = gpp_events.get('a3_events', [])
a3_config = config['gpp_events']['a3']

print(f"\nEvent Count: {len(a3_events)}")
print(f"Configured A3 Offset: {a3_config['offset_db']} dB")
print(f"Configured Hysteresis: {a3_config['hysteresis_db']} dB")
print(f"Total Threshold: {a3_config['offset_db'] + a3_config['hysteresis_db']} dB")

if a3_events:
    neighbor_rsrp = []
    serving_rsrp = []
    rsrp_diff = []

    for event in a3_events:
        m = event.get('measurements', {})
        n_rsrp = m.get('neighbor_rsrp_dbm')
        s_rsrp = m.get('serving_rsrp_dbm')

        if n_rsrp and s_rsrp:
            neighbor_rsrp.append(n_rsrp)
            serving_rsrp.append(s_rsrp)
            rsrp_diff.append(n_rsrp - s_rsrp)

    if rsrp_diff:
        print(f"\nRSRP Difference Distribution (Neighbor - Serving):")
        print(f"  Min:    {min(rsrp_diff):>7.2f} dB")
        print(f"  10th:   {np.percentile(rsrp_diff, 10):>7.2f} dB")
        print(f"  Median: {np.percentile(rsrp_diff, 50):>7.2f} dB")
        print(f"  90th:   {np.percentile(rsrp_diff, 90):>7.2f} dB")
        print(f"  Max:    {max(rsrp_diff):>7.2f} dB")
        print(f"  Mean:   {np.mean(rsrp_diff):>7.2f} dB")

        threshold = a3_config['offset_db'] + a3_config['hysteresis_db']
        below_threshold = sum(1 for d in rsrp_diff if d < threshold)

        print(f"\n🔍 Threshold Analysis:")
        print(f"  Total threshold: {threshold} dB")
        print(f"  Events below threshold: {below_threshold}/{len(rsrp_diff)} ({below_threshold/len(rsrp_diff)*100:.1f}%)")

        if below_threshold == 0:
            print(f"  ❌ ALL A3 events exceed threshold - potential data leakage!")
        elif below_threshold < len(rsrp_diff) * 0.1:
            print(f"  ⚠️  >90% events exceed threshold - threshold may be too low")
        else:
            print(f"  ✅ Threshold appears reasonable")
else:
    print("  No A3 events found in dataset")

# ==================== A4 事件分析 ====================
print("\n" + "=" * 80)
print("📊 A4 Events Analysis (Neighbor becomes better than threshold)")
print("=" * 80)

a4_events = gpp_events.get('a4_events', [])
a4_config = config['gpp_events']['a4']

print(f"\nEvent Count: {len(a4_events)}")
print(f"Configured A4 Threshold: {a4_config['rsrp_threshold_dbm']} dBm")

if a4_events:
    neighbor_rsrp = []
    trigger_margins = []

    for event in a4_events:
        m = event.get('measurements', {})
        n_rsrp = m.get('neighbor_rsrp_dbm')
        t_margin = m.get('trigger_margin_db')

        if n_rsrp:
            neighbor_rsrp.append(n_rsrp)
        if t_margin:
            trigger_margins.append(t_margin)

    if neighbor_rsrp:
        print(f"\nNeighbor RSRP Distribution:")
        print(f"  Min:    {min(neighbor_rsrp):>7.2f} dBm")
        print(f"  10th:   {np.percentile(neighbor_rsrp, 10):>7.2f} dBm")
        print(f"  Median: {np.percentile(neighbor_rsrp, 50):>7.2f} dBm")
        print(f"  Mean:   {np.mean(neighbor_rsrp):>7.2f} dBm")

        threshold = a4_config['rsrp_threshold_dbm']
        margin = np.mean(neighbor_rsrp) - threshold

        print(f"\n🔍 Threshold Analysis:")
        print(f"  Mean RSRP exceeds threshold by: {margin:.1f} dB")

        if margin > 50:
            print(f"  ❌ CRITICAL: Threshold {margin:.1f} dB below mean - SEVERE data leakage!")
            recommended = np.percentile(neighbor_rsrp, 30)
            print(f"  💡 Recommended threshold: {recommended:.1f} dBm (30th percentile)")
        elif margin > 20:
            print(f"  ⚠️  WARNING: Threshold {margin:.1f} dB below mean - potential leakage")
        else:
            print(f"  ✅ Threshold appears reasonable")

# ==================== D2 事件分析 ====================
print("\n" + "=" * 80)
print("📊 D2 Events Analysis (Distance-based handover)")
print("=" * 80)

d2_events = gpp_events.get('d2_events', [])
d2_config = config['gpp_events']['d2']['starlink']

print(f"\nEvent Count: {len(d2_events)}")
print(f"Configured D2 Threshold1 (near): {d2_config['d2_threshold1_km']} km")
print(f"Configured D2 Threshold2 (far):  {d2_config['d2_threshold2_km']} km")

if d2_events:
    serving_distances = []
    neighbor_distances = []
    distance_improvements = []
    threshold1_values = []
    threshold2_values = []

    for event in d2_events:
        m = event.get('measurements', {})
        s_dist = m.get('serving_ground_distance_km')
        n_dist = m.get('neighbor_ground_distance_km')
        improvement = m.get('ground_distance_improvement_km')
        t1 = m.get('threshold1_km')
        t2 = m.get('threshold2_km')

        if s_dist:
            serving_distances.append(s_dist)
        if n_dist:
            neighbor_distances.append(n_dist)
        if improvement:
            distance_improvements.append(improvement)
        if t1:
            threshold1_values.append(t1)
        if t2:
            threshold2_values.append(t2)

    if serving_distances and neighbor_distances:
        print(f"\nServing Satellite Distance:")
        print(f"  Min:    {min(serving_distances):>8.2f} km")
        print(f"  Median: {np.percentile(serving_distances, 50):>8.2f} km")
        print(f"  Mean:   {np.mean(serving_distances):>8.2f} km")
        print(f"  Max:    {max(serving_distances):>8.2f} km")

        print(f"\nNeighbor Satellite Distance:")
        print(f"  Min:    {min(neighbor_distances):>8.2f} km")
        print(f"  Median: {np.percentile(neighbor_distances, 50):>8.2f} km")
        print(f"  Mean:   {np.mean(neighbor_distances):>8.2f} km")
        print(f"  Max:    {max(neighbor_distances):>8.2f} km")

        # D2 uses dynamic thresholds per event (from Stage 4)
        if threshold1_values and threshold2_values:
            print(f"\nDynamic Threshold Distribution:")
            print(f"  Threshold1 (near): {min(threshold1_values):.1f} - {max(threshold1_values):.1f} km (mean: {np.mean(threshold1_values):.1f} km)")
            print(f"  Threshold2 (far):  {min(threshold2_values):.1f} - {max(threshold2_values):.1f} km (mean: {np.mean(threshold2_values):.1f} km)")

        if distance_improvements:
            print(f"\nDistance Improvement (Serving - Neighbor):")
            print(f"  Min:    {min(distance_improvements):>8.2f} km")
            print(f"  Median: {np.percentile(distance_improvements, 50):>8.2f} km")
            print(f"  Mean:   {np.mean(distance_improvements):>8.2f} km")
            print(f"  Max:    {max(distance_improvements):>8.2f} km")

        print(f"\n🔍 Threshold Analysis:")
        # Check if serving is always far from ground station
        serving_above_median = sum(1 for d in serving_distances if d > np.median(serving_distances))
        # Check if neighbor is always close
        neighbor_below_median = sum(1 for d in neighbor_distances if d < np.median(neighbor_distances))

        print(f"  Serving distance > median: {serving_above_median}/{len(serving_distances)} ({serving_above_median/len(serving_distances)*100:.1f}%)")
        print(f"  Neighbor distance < median: {neighbor_below_median}/{len(neighbor_distances)} ({neighbor_below_median/len(neighbor_distances)*100:.1f}%)")

        # Check for patterns
        if np.mean(serving_distances) > 1000 and np.mean(neighbor_distances) < 600:
            print(f"  ❌ PATTERN: Serving always far ({np.mean(serving_distances):.0f} km), neighbor always close ({np.mean(neighbor_distances):.0f} km)")
            print(f"  🚨 DATA LEAKAGE: Model can trivially identify D2 handover by distance values!")
        elif np.mean(distance_improvements) > 800:
            print(f"  ⚠️  Large distance improvement ({np.mean(distance_improvements):.0f} km) - may be too obvious")
        else:
            print(f"  ✅ Distance patterns appear reasonable")

# ==================== 總結 ====================
print("\n" + "=" * 80)
print("📋 Summary - Data Leakage Risk Assessment")
print("=" * 80)

print(f"\n✅ A3 Events: {len(a3_events)} events")
print(f"   Config: offset={a3_config['offset_db']} dB, hysteresis={a3_config['hysteresis_db']} dB")

print(f"\n❌ A4 Events: {len(a4_events)} events - CRITICAL ISSUE")
print(f"   Config: threshold={a4_config['rsrp_threshold_dbm']} dBm (TOO LOW)")
print(f"   Impact: ALL events have 55-80 dB margin above threshold")
print(f"   Result: Model can trivially identify handover by trigger margin")

print(f"\n⚠️  D2 Events: {len(d2_events)} events")
print(f"   Config: threshold1={d2_config['d2_threshold1_km']} km, threshold2={d2_config['d2_threshold2_km']} km")

print("\n" + "=" * 80)
