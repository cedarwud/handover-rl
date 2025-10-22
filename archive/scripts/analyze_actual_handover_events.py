#!/usr/bin/env python3
"""
Analyze ACTUAL Handover Events from Stage 6

Extract threshold recommendations from REAL handover events that already occurred.
This is the ground truth - these events represent what the system considered
worthy of handover.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from adapters.handover_event_loader import create_handover_event_loader

print("=" * 80)
print("THRESHOLD ANALYSIS FROM ACTUAL HANDOVER EVENTS")
print("(Ground Truth: Events that system already triggered)")
print("=" * 80)

# Load actual handover events
loader = create_handover_event_loader()
orbit_engine_root = Path(__file__).parent.parent / 'orbit-engine'
stage6_dir = orbit_engine_root / 'data' / 'outputs' / 'rl_training' / 'stage6'

a4_events, d2_events = loader.load_latest_events(stage6_dir)

# Also load Stage 5 to get serving satellite RSRP
stage5_dir = orbit_engine_root / 'data' / 'outputs' / 'rl_training' / 'stage5'
stage5_file = sorted(stage5_dir.glob('stage5_signal_analysis_*.json'))[-1]

with open(stage5_file, 'r') as f:
    stage5_data = json.load(f)

signal_analysis = stage5_data['signal_analysis']

print(f"\n✅ Loaded:")
print(f"   A4 Events: {len(a4_events)}")
print(f"   D2 Events: {len(d2_events)}")
print(f"   Stage 5 satellites: {len(signal_analysis)}")

# ==================== ANALYSIS 1: A4 Events ====================
print("\n" + "=" * 80)
print("📊 ANALYSIS 1: A4 Events (Neighbor exceeds threshold)")
print("=" * 80)

a4_neighbor_rsrp = []
a4_trigger_margins = []
a4_thresholds_used = []

for event in a4_events:
    measurements = event.get('measurements', {})

    neighbor_rsrp = measurements.get('neighbor_rsrp_dbm')
    threshold = measurements.get('threshold_dbm')
    trigger_margin = measurements.get('trigger_margin_db')

    if neighbor_rsrp is not None:
        a4_neighbor_rsrp.append(neighbor_rsrp)

    if threshold is not None:
        a4_thresholds_used.append(threshold)

    if trigger_margin is not None:
        a4_trigger_margins.append(trigger_margin)

print(f"\n📈 A4 Neighbor RSRP Distribution (N={len(a4_neighbor_rsrp)}):")
print(f"   Min:        {np.min(a4_neighbor_rsrp):>7.2f} dBm")
print(f"   10th %ile:  {np.percentile(a4_neighbor_rsrp, 10):>7.2f} dBm")
print(f"   25th %ile:  {np.percentile(a4_neighbor_rsrp, 25):>7.2f} dBm")
print(f"   Median:     {np.percentile(a4_neighbor_rsrp, 50):>7.2f} dBm")
print(f"   75th %ile:  {np.percentile(a4_neighbor_rsrp, 75):>7.2f} dBm")
print(f"   90th %ile:  {np.percentile(a4_neighbor_rsrp, 90):>7.2f} dBm")
print(f"   Max:        {np.max(a4_neighbor_rsrp):>7.2f} dBm")
print(f"   Mean:       {np.mean(a4_neighbor_rsrp):>7.2f} dBm")

print(f"\n📈 A4 Threshold Used (what was configured):")
print(f"   Unique thresholds: {set(a4_thresholds_used)}")
print(f"   Most common: {np.median(a4_thresholds_used):.2f} dBm")

print(f"\n📈 A4 Trigger Margin (neighbor_rsrp - threshold):")
print(f"   Min:        {np.min(a4_trigger_margins):>7.2f} dB")
print(f"   10th %ile:  {np.percentile(a4_trigger_margins, 10):>7.2f} dB")
print(f"   Median:     {np.percentile(a4_trigger_margins, 50):>7.2f} dB")
print(f"   90th %ile:  {np.percentile(a4_trigger_margins, 90):>7.2f} dB")
print(f"   Max:        {np.max(a4_trigger_margins):>7.2f} dB")
print(f"   Mean:       {np.mean(a4_trigger_margins):>7.2f} dB")

# Key insight: What should threshold be to create realistic margins?
print(f"\n💡 A4 Threshold Recommendation:")
print(f"   Current threshold: {np.median(a4_thresholds_used):.2f} dBm")
print(f"   Neighbor RSRP range: {np.min(a4_neighbor_rsrp):.2f} to {np.max(a4_neighbor_rsrp):.2f} dBm")
print(f"   ")
print(f"   To create realistic margins (-5 to +10 dB), threshold should be:")

# Calculate what threshold would give realistic margins
target_margin_low = -5  # Some neighbors below threshold
target_margin_high = 10  # Some neighbors above threshold

# Threshold that puts ~30% of neighbors below it
recommended_a4 = np.percentile(a4_neighbor_rsrp, 30)
margins_with_new_threshold = [rsrp - recommended_a4 for rsrp in a4_neighbor_rsrp]

print(f"   Recommended: {recommended_a4:.1f} dBm (30th percentile of neighbor RSRP)")
print(f"   This would create margins: {np.min(margins_with_new_threshold):.1f} to {np.max(margins_with_new_threshold):.1f} dB")
print(f"   Negative margins: {sum(1 for m in margins_with_new_threshold if m < 0)}/{len(margins_with_new_threshold)} ({sum(1 for m in margins_with_new_threshold if m < 0)/len(margins_with_new_threshold)*100:.1f}%)")

# ==================== ANALYSIS 2: D2 Events ====================
print("\n" + "=" * 80)
print("📊 ANALYSIS 2: D2 Events (Distance-based)")
print("=" * 80)

d2_serving_distances = []
d2_neighbor_distances = []
d2_distance_improvements = []

for event in d2_events:
    measurements = event.get('measurements', {})

    serving_dist = measurements.get('serving_ground_distance_km')
    neighbor_dist = measurements.get('neighbor_ground_distance_km')
    improvement = measurements.get('ground_distance_improvement_km')

    if serving_dist:
        d2_serving_distances.append(serving_dist)
    if neighbor_dist:
        d2_neighbor_distances.append(neighbor_dist)
    if improvement:
        d2_distance_improvements.append(improvement)

if d2_serving_distances:
    print(f"\n📈 D2 Serving Distance (N={len(d2_serving_distances)}):")
    print(f"   Min:     {np.min(d2_serving_distances):>8.2f} km")
    print(f"   Median:  {np.percentile(d2_serving_distances, 50):>8.2f} km")
    print(f"   Mean:    {np.mean(d2_serving_distances):>8.2f} km")
    print(f"   Max:     {np.max(d2_serving_distances):>8.2f} km")

if d2_neighbor_distances:
    print(f"\n📈 D2 Neighbor Distance (N={len(d2_neighbor_distances)}):")
    print(f"   Min:     {np.min(d2_neighbor_distances):>8.2f} km")
    print(f"   Median:  {np.percentile(d2_neighbor_distances, 50):>8.2f} km")
    print(f"   Mean:    {np.mean(d2_neighbor_distances):>8.2f} km")
    print(f"   Max:     {np.max(d2_neighbor_distances):>8.2f} km")

if d2_distance_improvements:
    print(f"\n📈 D2 Distance Improvement (N={len(d2_distance_improvements)}):")
    print(f"   Min:     {np.min(d2_distance_improvements):>8.2f} km")
    print(f"   Median:  {np.percentile(d2_distance_improvements, 50):>8.2f} km")
    print(f"   Mean:    {np.mean(d2_distance_improvements):>8.2f} km")
    print(f"   Max:     {np.max(d2_distance_improvements):>8.2f} km")

    print(f"\n💡 D2 Threshold Recommendations:")
    # Threshold1: serving must exceed this (should be < median serving distance)
    recommended_d2_t1 = np.percentile(d2_serving_distances, 40)
    print(f"   Threshold1 (serving > X): {recommended_d2_t1:.1f} km (40th percentile)")

    # Threshold2: neighbor must be below this (should be > median neighbor distance)
    recommended_d2_t2 = np.percentile(d2_neighbor_distances, 70)
    print(f"   Threshold2 (neighbor < X): {recommended_d2_t2:.1f} km (70th percentile)")

    print(f"   This would satisfy: ~40% of serving and ~70% of neighbors")

# ==================== ANALYSIS 3: Get Serving RSRP from Stage 5 ====================
print("\n" + "=" * 80)
print("📊 ANALYSIS 3: Serving Satellite RSRP (from Stage 5)")
print("=" * 80)

# Try to get serving satellite RSRP for A4 events
serving_rsrp_list = []
rsrp_differences = []

print(f"\n🔍 Extracting serving satellite RSRP for A4 events...")

for event in a4_events[:1000]:  # Sample first 1000
    serving_sat_id = event.get('serving_satellite')
    timestamp = event.get('timestamp')
    neighbor_rsrp = event.get('measurements', {}).get('neighbor_rsrp_dbm')

    if not serving_sat_id or not timestamp or neighbor_rsrp is None:
        continue

    # Look up serving satellite in Stage 5
    if serving_sat_id in signal_analysis:
        sat_data = signal_analysis[serving_sat_id]
        time_series = sat_data.get('time_series', [])

        # Find closest time point
        for tp in time_series:
            tp_timestamp = tp.get('timestamp')
            if tp_timestamp and abs((tp_timestamp - timestamp).total_seconds()) < 60:
                signal_quality = tp.get('signal_quality', {})
                serving_rsrp = signal_quality.get('rsrp_dbm')

                if serving_rsrp:
                    serving_rsrp_list.append(serving_rsrp)
                    rsrp_diff = neighbor_rsrp - serving_rsrp
                    rsrp_differences.append(rsrp_diff)
                break

print(f"✅ Matched {len(serving_rsrp_list)} events with serving RSRP")

if rsrp_differences:
    print(f"\n📈 RSRP Difference (Neighbor - Serving) [N={len(rsrp_differences)}]:")
    print(f"   Min:        {np.min(rsrp_differences):>7.2f} dB")
    print(f"   10th %ile:  {np.percentile(rsrp_differences, 10):>7.2f} dB")
    print(f"   25th %ile:  {np.percentile(rsrp_differences, 25):>7.2f} dB")
    print(f"   Median:     {np.percentile(rsrp_differences, 50):>7.2f} dB")
    print(f"   75th %ile:  {np.percentile(rsrp_differences, 75):>7.2f} dB")
    print(f"   90th %ile:  {np.percentile(rsrp_differences, 90):>7.2f} dB")
    print(f"   Max:        {np.max(rsrp_differences):>7.2f} dB")
    print(f"   Mean:       {np.mean(rsrp_differences):>7.2f} dB")

    print(f"\n💡 A3 Offset Recommendation:")
    # A3 offset should be the typical RSRP difference that justifies handover
    # Use median or a percentile
    recommended_a3_offset = np.percentile(rsrp_differences, 50)
    print(f"   Based on actual handover RSRP differences:")
    print(f"   Recommended A3 offset: {recommended_a3_offset:.1f} dB (median of actual events)")
    print(f"   This represents the typical gain when handover occurs")

# ==================== FINAL SUMMARY ====================
print("\n" + "=" * 80)
print("🎯 FINAL RECOMMENDATIONS (Based on ACTUAL Handover Events)")
print("=" * 80)

print(f"\n```yaml")
print(f"gpp_events:")
print(f"  a3:")
if rsrp_differences:
    print(f"    offset_db: {np.percentile(rsrp_differences, 50):.1f}  # Median RSRP diff in actual handovers")
else:
    print(f"    offset_db: 3.0  # Default (no serving RSRP data)")
print(f"    hysteresis_db: 1.5")
print(f"")
print(f"  a4:")
print(f"    rsrp_threshold_dbm: {recommended_a4:.1f}  # 30th percentile of neighbor RSRP")
print(f"    hysteresis_db: 2.0")
print(f"")
print(f"  a5:")
print(f"    rsrp_threshold1_dbm: {np.percentile(a4_neighbor_rsrp, 10):.1f}  # 10th percentile")
print(f"    rsrp_threshold2_dbm: {np.percentile(a4_neighbor_rsrp, 40):.1f}  # 40th percentile")
print(f"    hysteresis_db: 2.0")
if d2_serving_distances:
    print(f"")
    print(f"  d2:")
    print(f"    starlink:")
    print(f"      d2_threshold1_km: {np.percentile(d2_serving_distances, 40):.1f}  # 40th percentile serving")
    print(f"      d2_threshold2_km: {np.percentile(d2_neighbor_distances, 70):.1f}  # 70th percentile neighbor")
print(f"```")

print("\n" + "=" * 80)
