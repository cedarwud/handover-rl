#!/usr/bin/env python3
"""
Visibility Pattern Diagnostic Tool

Analyzes satellite visibility patterns over time to understand
why handover rate is 0%.

Purpose:
- Sample visibility at different times
- Identify periods of high/low satellite coverage
- Validate episode temporal sampling strategy
"""

import sys
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from adapters.orbit_engine_adapter import OrbitEngineAdapter
from utils.satellite_utils import load_stage4_optimized_satellites

def check_visibility_at_time(adapter, satellite_ids, timestamp, min_elevation=10.0):
    """Check how many satellites are visible at a specific time"""
    visible_count = 0
    visible_sats = []

    for sat_id in satellite_ids:
        try:
            state_dict = adapter.calculate_state(
                satellite_id=sat_id,
                timestamp=timestamp
            )

            if not state_dict:
                continue

            elevation = state_dict.get('elevation_deg', 0)
            if elevation < min_elevation:
                continue

            if not state_dict.get('is_connectable', False):
                continue

            visible_count += 1
            visible_sats.append({
                'id': sat_id,
                'elevation': elevation,
                'rsrp': state_dict['rsrp_dbm'],
                'distance': state_dict.get('distance_km', 0)
            })
        except Exception as e:
            continue

    return visible_count, visible_sats


def diagnose_visibility_pattern(
    adapter,
    satellite_ids,
    start_time,
    duration_hours=24,
    sample_interval_minutes=10
):
    """
    Sample visibility over time to find patterns

    Args:
        adapter: OrbitEngineAdapter
        satellite_ids: List of satellite IDs
        start_time: Start of sampling window
        duration_hours: How long to sample
        sample_interval_minutes: Time between samples
    """
    print(f"\n{'='*80}")
    print(f"VISIBILITY PATTERN DIAGNOSTIC")
    print(f"{'='*80}")
    print(f"Start time: {start_time}")
    print(f"Duration: {duration_hours} hours")
    print(f"Sample interval: {sample_interval_minutes} minutes")
    print(f"Total satellites: {len(satellite_ids)}")
    print(f"{'='*80}\n")

    # Sample visibility
    num_samples = int(duration_hours * 60 / sample_interval_minutes)
    visibility_counts = []
    timestamps = []
    visibility_details = []

    print("Sampling visibility...")
    for i in range(num_samples):
        offset_minutes = i * sample_interval_minutes
        sample_time = start_time + timedelta(minutes=offset_minutes)

        count, sats = check_visibility_at_time(adapter, satellite_ids, sample_time)
        visibility_counts.append(count)
        timestamps.append(sample_time)
        visibility_details.append(sats)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{num_samples} samples")

    # Statistics
    visibility_array = np.array(visibility_counts)

    print(f"\n{'='*80}")
    print(f"VISIBILITY STATISTICS")
    print(f"{'='*80}")
    print(f"Min visible satellites: {visibility_array.min()}")
    print(f"Max visible satellites: {visibility_array.max()}")
    print(f"Mean visible satellites: {visibility_array.mean():.2f}")
    print(f"Std visible satellites: {visibility_array.std():.2f}")
    print(f"Median visible satellites: {np.median(visibility_array):.1f}")

    # Distribution
    print(f"\nVisibility Distribution:")
    unique, counts = np.unique(visibility_array, return_counts=True)
    for vis, cnt in zip(unique, counts):
        pct = cnt / len(visibility_array) * 100
        print(f"  {int(vis)} satellites: {cnt:3d} samples ({pct:5.1f}%)")

    # Find best periods (>=4 satellites for meaningful handover)
    good_periods = []
    for i, (ts, count) in enumerate(zip(timestamps, visibility_counts)):
        if count >= 4:
            good_periods.append((ts, count))

    print(f"\n{'='*80}")
    print(f"HIGH COVERAGE PERIODS (>=4 satellites)")
    print(f"{'='*80}")
    print(f"Found {len(good_periods)} samples with >=4 satellites")
    print(f"Coverage: {len(good_periods)/len(timestamps)*100:.1f}% of sampled time")

    if len(good_periods) > 0:
        print(f"\nFirst 10 high-coverage periods:")
        for ts, count in good_periods[:10]:
            print(f"  {ts.strftime('%Y-%m-%d %H:%M:%S')}: {count} satellites")

        # Recommend episode start times
        print(f"\n{'='*80}")
        print(f"RECOMMENDED EPISODE START TIMES")
        print(f"{'='*80}")
        print(f"Use these timestamps for evaluation episodes:")
        for i, (ts, count) in enumerate(good_periods[:5]):
            print(f"  Episode {i}: {ts.isoformat()} ({count} satellites)")
    else:
        print(f"\n⚠️  NO high-coverage periods found!")
        print(f"    Consider:")
        print(f"    1. Expanding time window")
        print(f"    2. Using more satellites")
        print(f"    3. Lowering min_elevation threshold")

    # Handover potential analysis
    print(f"\n{'='*80}")
    print(f"HANDOVER POTENTIAL ANALYSIS")
    print(f"{'='*80}")

    # Simulate 95-minute episode at different start times
    episode_duration_minutes = 95
    episode_samples = int(episode_duration_minutes / sample_interval_minutes)

    handover_potential_episodes = []
    for i in range(0, len(timestamps) - episode_samples, episode_samples):
        episode_visibility = visibility_counts[i:i+episode_samples]
        episode_start = timestamps[i]

        # Check if episode has potential for handovers
        min_vis = min(episode_visibility)
        max_vis = max(episode_visibility)
        mean_vis = np.mean(episode_visibility)

        # Handover potential: need at least 2 satellites and variation
        has_potential = (min_vis >= 2) and (max_vis >= 3)

        if has_potential:
            handover_potential_episodes.append({
                'start': episode_start,
                'min_vis': min_vis,
                'max_vis': max_vis,
                'mean_vis': mean_vis,
            })

    print(f"Episodes with handover potential (min>=2, max>=3):")
    print(f"  Total: {len(handover_potential_episodes)}")

    if len(handover_potential_episodes) > 0:
        print(f"\nTop 5 episodes by mean visibility:")
        sorted_episodes = sorted(handover_potential_episodes, key=lambda x: x['mean_vis'], reverse=True)
        for i, ep in enumerate(sorted_episodes[:5]):
            print(f"  {i+1}. Start: {ep['start'].strftime('%Y-%m-%d %H:%M')}, "
                  f"Visibility: {ep['min_vis']:.0f}-{ep['max_vis']:.0f} "
                  f"(avg {ep['mean_vis']:.1f})")
    else:
        print(f"\n⚠️  NO episodes with handover potential found!")
        print(f"    This explains the 0% handover rate.")

    return {
        'timestamps': timestamps,
        'visibility_counts': visibility_counts,
        'visibility_details': visibility_details,
        'good_periods': good_periods,
        'handover_potential_episodes': handover_potential_episodes,
    }


def main():
    print("\n" + "="*80)
    print("LEO SATELLITE VISIBILITY DIAGNOSTIC")
    print("="*80)

    # Load config
    config_path = Path(__file__).parent.parent.parent / 'config' / 'data_gen_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize adapter
    print("\nInitializing OrbitEngineAdapter...")
    adapter = OrbitEngineAdapter(config)

    # Load satellites
    print("Loading Stage 4 satellite pool...")
    satellite_ids, metadata = load_stage4_optimized_satellites(
        constellation_filter='starlink',
        return_metadata=True
    )

    print(f"  Total satellites: {len(satellite_ids)}")

    # Use TLE epoch time (2025-10-18) as base
    start_time = datetime(2025, 10, 18, 0, 0, 0)

    # Run diagnostic
    results = diagnose_visibility_pattern(
        adapter=adapter,
        satellite_ids=satellite_ids,
        start_time=start_time,
        duration_hours=24,
        sample_interval_minutes=5
    )

    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
