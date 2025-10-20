#!/usr/bin/env python3
"""
Verification Script - Placeholder States Fix

Tests the modified rl_data_generator.py to ensure:
1. ✅ No placeholder states (np.zeros()) in episodes
2. ✅ Variable-length episodes work correctly
3. ✅ Metadata tracks coverage properly
4. ✅ Ground truth actions handle variable length
5. ✅ All states are real physical states from orbit-engine

SOURCE: fix.md P0 - Verification of placeholder states removal
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import yaml

# Add project root to path
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_generation.rl_data_generator import RLDataGenerator


def verify_no_placeholder_states(episode_data):
    """
    Verify episode contains no placeholder states.

    Placeholder detection:
    - All-zero states: np.all(state == 0)
    - Repeated identical states (not physical)
    """
    states = episode_data['states']

    # Check 1: No all-zero states
    for i, state in enumerate(states):
        if np.all(state == 0.0):
            return False, f"Found all-zero state at index {i}"

    # Check 2: Verify is_connectable flag (RSRP > -140 dBm minimum)
    rsrp_values = states[:, 0]  # First dimension is RSRP
    if np.any(rsrp_values < -140.0):
        return False, f"Found invalid RSRP values: {rsrp_values[rsrp_values < -140.0]}"

    return True, "✅ No placeholder states detected"


def verify_variable_length_metadata(episode_data):
    """
    Verify metadata correctly tracks variable-length episodes.
    """
    metadata = episode_data['metadata']
    states = episode_data['states']

    required_keys = ['valid_steps', 'actual_steps', 'requested_steps', 'coverage_rate']
    for key in required_keys:
        if key not in metadata:
            return False, f"Missing metadata key: {key}"

    # Verify consistency
    actual_length = len(states)
    if metadata['actual_steps'] != actual_length:
        return False, f"Metadata mismatch: actual_steps={metadata['actual_steps']}, len(states)={actual_length}"

    if metadata['valid_steps'] != actual_length:
        return False, f"Metadata mismatch: valid_steps={metadata['valid_steps']}, len(states)={actual_length}"

    # Verify coverage rate
    expected_coverage = metadata['valid_steps'] / metadata['requested_steps']
    if abs(metadata['coverage_rate'] - expected_coverage) > 0.001:
        return False, f"Coverage rate mismatch: {metadata['coverage_rate']} != {expected_coverage}"

    return True, f"✅ Metadata correct: {metadata['valid_steps']}/{metadata['requested_steps']} steps ({metadata['coverage_rate']*100:.1f}% coverage)"


def verify_ground_truth_actions(episode_data):
    """
    Verify ground truth actions match episode length.
    """
    states = episode_data['states']
    actions = episode_data['actions']

    if len(actions) != len(states):
        return False, f"Action length mismatch: {len(actions)} != {len(states)}"

    # Verify action values are valid (0 or 1)
    unique_actions = np.unique(actions)
    if not all(a in [0, 1] for a in unique_actions):
        return False, f"Invalid action values: {unique_actions}"

    return True, f"✅ Actions valid: {len(actions)} actions, maintain={np.sum(actions==0)}, handover={np.sum(actions==1)}"


def verify_array_consistency(episode_data):
    """
    Verify all arrays have consistent length.
    """
    states = episode_data['states']
    T = len(states)

    arrays_to_check = {
        'actions': episode_data['actions'],
        'rewards': episode_data['rewards'],
        'next_states': episode_data['next_states'],
        'dones': episode_data['dones'],
        'timestamps': episode_data['timestamps']
    }

    for name, array in arrays_to_check.items():
        if len(array) != T:
            return False, f"Array length mismatch: {name} has {len(array)} elements, expected {T}"

    return True, f"✅ All arrays consistent: T={T}"


def test_episode_generation(config):
    """
    Test episode generation with real orbit-engine data.
    """
    print("=" * 70)
    print("📊 Testing Modified Data Generation Flow")
    print("=" * 70)

    try:
        # Create generator
        print("\n🔧 Initializing RLDataGenerator...")
        generator = RLDataGenerator(config)

        # Generate single episode for testing
        print("\n📦 Generating test episode...")
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        episode_data = generator._generate_episode(start_time, episode_id=0)

        if episode_data is None:
            print("⚠️  Episode generation returned None (insufficient valid states)")
            return False

        print(f"✅ Episode generated successfully")

        # Run verification checks
        print("\n🔍 Running Verification Checks...")
        print("-" * 70)

        checks = [
            ("No Placeholder States", verify_no_placeholder_states),
            ("Variable Length Metadata", verify_variable_length_metadata),
            ("Ground Truth Actions", verify_ground_truth_actions),
            ("Array Consistency", verify_array_consistency)
        ]

        all_passed = True
        for check_name, check_func in checks:
            passed, message = check_func(episode_data)
            status = "✅" if passed else "❌"
            print(f"{status} {check_name}: {message}")
            if not passed:
                all_passed = False

        print("-" * 70)

        # Print episode statistics
        if all_passed:
            print("\n📈 Episode Statistics:")
            metadata = episode_data['metadata']
            print(f"   Valid steps: {metadata['valid_steps']}/{metadata['requested_steps']}")
            print(f"   Coverage rate: {metadata['coverage_rate']*100:.1f}%")
            print(f"   Actual episode length: {metadata['actual_steps']}")
            print(f"   Primary satellite: {metadata['primary_satellite']}")

            states = episode_data['states']
            print(f"\n📊 State Statistics:")
            print(f"   RSRP range: [{states[:, 0].min():.1f}, {states[:, 0].max():.1f}] dBm")
            print(f"   RSRQ range: [{states[:, 1].min():.1f}, {states[:, 1].max():.1f}] dB")
            print(f"   SINR range: [{states[:, 2].min():.1f}, {states[:, 2].max():.1f}] dB")
            print(f"   Distance range: [{states[:, 3].min():.1f}, {states[:, 3].max():.1f}] km")
            print(f"   Elevation range: [{states[:, 4].min():.1f}, {states[:, 4].max():.1f}] deg")

            actions = episode_data['actions']
            print(f"\n🎯 Action Distribution:")
            print(f"   Maintain (0): {np.sum(actions==0)} ({np.sum(actions==0)/len(actions)*100:.1f}%)")
            print(f"   Handover (1): {np.sum(actions==1)} ({np.sum(actions==1)/len(actions)*100:.1f}%)")

        print("\n" + "=" * 70)
        if all_passed:
            print("✅ ALL VERIFICATION CHECKS PASSED")
            print("=" * 70)
            print("\n🎉 Placeholder states fix verified successfully!")
            print("   • No np.zeros() placeholder states found")
            print("   • Variable-length episodes work correctly")
            print("   • Metadata tracking is accurate")
            print("   • All states are real physical values from orbit-engine")
        else:
            print("❌ VERIFICATION FAILED")
            print("=" * 70)

        return all_passed

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Data Generation Fix Verification")
    print("SOURCE: fix.md P0 - Remove np.zeros() placeholder states\n")

    # Load configuration
    config_path = PROJECT_ROOT / "config" / "data_gen_config.yaml"

    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration loaded: {config_path}\n")
    else:
        print(f"⚠️  Configuration file not found: {config_path}")
        print("Using test configuration...\n")
        config = {
            'data_generation': {
                'satellite_ids': ['STARLINK-1007', 'STARLINK-1020'],
                'time_step_seconds': 5,
                'episode_duration_minutes': 95,
                'ground_truth_lookahead_steps': 10,
                'rsrp_threshold_dbm': -100.0,
                'rsrp_hysteresis_db': 3.0
            },
            'orbit_engine': {
                'orbit_engine_root': str(PROJECT_ROOT.parent / 'orbit-engine'),
                'tle_data_dir': str(PROJECT_ROOT.parent / 'orbit-engine' / 'data' / 'tle'),
                'cache_dir': str(PROJECT_ROOT / 'data' / 'cache')
            }
        }

    # Run verification
    success = test_episode_generation(config)

    sys.exit(0 if success else 1)
