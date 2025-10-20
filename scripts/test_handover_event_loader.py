#!/usr/bin/env python3
"""
Test Handover Event Loader - Verify A4/D2 Event Loading

Tests:
1. Load events from Stage 6 output
2. Validate event structure
3. Extract baseline policy
4. Display statistics

Usage:
    python scripts/test_handover_event_loader.py
"""

import sys
from pathlib import Path

# Import directly from module file to avoid __init__.py
handover_loader_path = Path(__file__).parent.parent / 'src' / 'adapters'
sys.path.insert(0, str(handover_loader_path))

from handover_event_loader import create_handover_event_loader
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

def main():
    """Test handover event loader."""
    print("=" * 80)
    print("Testing Handover Event Loader")
    print("=" * 80)

    # Create loader
    loader = create_handover_event_loader()

    # Find Stage 6 output directory
    orbit_engine_root = Path(__file__).parent.parent.parent / 'orbit-engine'
    stage6_dir = orbit_engine_root / 'data' / 'outputs' / 'rl_training' / 'stage6'

    print(f"\n📂 Looking for Stage 6 outputs in:")
    print(f"   {stage6_dir}")

    if not stage6_dir.exists():
        print(f"\n❌ Error: Stage 6 output directory not found")
        print(f"   Please run orbit-engine Stage 6 first:")
        print(f"   cd {orbit_engine_root}")
        print(f"   ./test_rl_small.sh")
        return 1

    # Load events
    try:
        a4_events, d2_events = loader.load_latest_events(stage6_dir)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return 1
    except ValueError as e:
        print(f"\n❌ Validation Error: {e}")
        return 1

    # Display results
    print(f"\n✅ Successfully loaded handover events:")
    print(f"   - A4 events: {len(a4_events)}")
    print(f"   - D2 events: {len(d2_events)}")
    print(f"   - Total: {len(a4_events) + len(d2_events)}")

    # Display A4 event example
    if a4_events:
        print(f"\n📊 A4 Event Example (Neighbour better than threshold):")
        event = a4_events[0]
        print(f"   Event Type: {event['event_type']}")
        print(f"   Timestamp: {event['timestamp']}")
        print(f"   Serving Satellite: {event['serving_satellite']}")
        print(f"   Neighbor Satellite: {event['neighbor_satellite']}")
        measurements = event['measurements']
        print(f"   Neighbor RSRP: {measurements['neighbor_rsrp_dbm']:.2f} dBm")
        print(f"   Threshold: {measurements['threshold_dbm']:.2f} dBm")
        print(f"   Margin: {measurements['trigger_margin_db']:.2f} dB (neighbor better by this amount)")
        print(f"   Standard: {event.get('standard_reference', 'N/A')}")

    # Display D2 event example
    if d2_events:
        print(f"\n📊 D2 Event Example:")
        event = d2_events[0]
        print(f"   Event Type: {event['event_type']}")
        print(f"   Timestamp: {event['timestamp']}")
        print(f"   Serving Satellite: {event['serving_satellite']}")
        print(f"   Neighbor Satellite: {event['neighbor_satellite']}")
        measurements = event['measurements']
        print(f"   Serving Distance: {measurements['serving_ground_distance_km']:.1f} km")
        print(f"   Neighbor Distance: {measurements['neighbor_ground_distance_km']:.1f} km")
        print(f"   Improvement: {measurements['serving_ground_distance_km'] - measurements['neighbor_ground_distance_km']:.1f} km")
        print(f"   Standard: {event.get('standard_reference', 'N/A')}")

    # Extract baseline policy
    print(f"\n🎯 Extracting baseline handover policy...")
    policy = loader.extract_baseline_policy(a4_events, d2_events)

    print(f"\n📋 Baseline Policy Summary:")
    print(f"   Total Events: {policy['total_events']}")

    # A4 Policy
    a4_policy = policy['a4_policy']
    if a4_policy['enabled']:
        print(f"\n   A4 Policy (Threshold-based):")
        print(f"     ✅ Enabled: {a4_policy['event_count']} events")
        neighbor_stats = a4_policy['neighbor_rsrp_statistics']
        threshold_stats = a4_policy['threshold_statistics']
        print(f"     Neighbor RSRP Range: {neighbor_stats['min']:.2f} ~ {neighbor_stats['max']:.2f} dBm")
        print(f"     Threshold Range: {threshold_stats['min']:.2f} ~ {threshold_stats['max']:.2f} dBm")
        print(f"     Recommended Threshold: {a4_policy['recommended_threshold_dbm']:.2f} dBm (median)")
        print(f"     Standard: {a4_policy['standard_reference']}")
    else:
        print(f"\n   A4 Policy: ❌ Disabled ({a4_policy['reason']})")

    # D2 Policy
    d2_policy = policy['d2_policy']
    if d2_policy['enabled']:
        print(f"\n   D2 Policy (Distance-based):")
        print(f"     ✅ Enabled: {d2_policy['event_count']} events")
        stats = d2_policy['distance_statistics']
        print(f"     Avg Serving Distance: {stats['serving_mean_km']:.1f} km")
        print(f"     Avg Neighbor Distance: {stats['neighbor_mean_km']:.1f} km")
        print(f"     Avg Improvement: {stats['improvement_mean_km']:.1f} km")
        print(f"     Standard: {d2_policy['standard_reference']}")
    else:
        print(f"\n   D2 Policy: ❌ Disabled ({d2_policy['reason']})")

    print(f"\n" + "=" * 80)
    print(f"✅ Handover Event Loader Test Passed!")
    print(f"=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
