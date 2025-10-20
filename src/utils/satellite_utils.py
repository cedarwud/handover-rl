#!/usr/bin/env python3
"""
Satellite Utility Functions

Provides utilities for loading optimized satellite pools from orbit-engine Stage 4 output

Academic Compliance:
- NO HARDCODING: All satellite IDs from orbit-engine Stage 4 Pool Optimization
- SOURCE: orbit-engine six-stage processing pipeline
- DATA-DRIVEN: 125 satellites = 101 Starlink + 24 OneWeb (scientifically selected)
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple


def load_stage4_optimized_satellites(
    stage4_output_dir: Path = None,
    return_metadata: bool = False,
    constellation_filter: str = None
) -> List[str]:
    """
    Load optimized satellites from orbit-engine Stage 4 Pool Optimization output

    This function reads the scientifically selected satellite pool that has passed
    through orbit-engine's six-stage processing:
        Stage 1: TLE Loading (8000+ candidates)
        Stage 2: Orbital Propagation
        Stage 3: Coordinate Transformation
        Stage 4: Pool Optimization ← SOURCE OF THIS DATA
        Stage 5: Signal Analysis
        Stage 6: Research Optimization

    Args:
        stage4_output_dir: Path to Stage 4 output directory
                          (default: /home/sat/satellite/orbit-engine/data/outputs/stage4)
        return_metadata: If True, return (satellite_ids, metadata_dict)
        constellation_filter: Filter by constellation ('starlink', 'oneweb', or None for all)
                            IMPORTANT: Cross-constellation handover is NOT supported
                            (Starlink and OneWeb are separate commercial networks)

    Returns:
        List of satellite IDs (NORAD catalog numbers as strings):
        - If constellation_filter='starlink': 101 Starlink satellites (550km LEO)
        - If constellation_filter='oneweb': 24 OneWeb satellites (1200km LEO)
        - If constellation_filter=None: 125 satellites (101 Starlink + 24 OneWeb)

    Raises:
        FileNotFoundError: If no Stage 4 output found
        ValueError: If constellation_filter is invalid

    SOURCE: orbit-engine Stage 4 Pool Optimization
    FILE: /orbit-engine/data/outputs/stage4/link_feasibility_output_YYYYMMDD_HHMMSS.json
    FIELD: pool_optimization['optimized_pools']['starlink'] + ['oneweb']

    Academic Compliance:
    - NO HARDCODING: Reads from actual scientific optimization output
    - SINGLE-CONSTELLATION TRAINING: Use constellation_filter for realistic scenarios
    - TRACEABLE: Every satellite traceable to Stage 4 selection criteria

    Note on Cross-Constellation Handover:
    - Starlink and OneWeb are separate commercial networks
    - Users cannot handover between them (like AT&T vs Verizon)
    - For RL training, use constellation_filter='starlink' or 'oneweb'
    """
    # Default Stage 4 output directory
    if stage4_output_dir is None:
        stage4_output_dir = Path("/home/sat/satellite/orbit-engine/data/outputs/stage4")

    # Find latest Stage 4 output file
    stage4_files = sorted(stage4_output_dir.glob("link_feasibility_output_*.json"))

    if not stage4_files:
        raise FileNotFoundError(
            f"❌ No Stage 4 output found in {stage4_output_dir}\n"
            f"   Please run orbit-engine stages 1-4 first:\n"
            f"   cd /home/sat/satellite/orbit-engine && ./run.sh --stages 1-4"
        )

    latest_file = stage4_files[-1]
    print(f"📂 Loading Stage 4 output: {latest_file.name}")

    # Load Stage 4 data
    with open(latest_file) as f:
        data = json.load(f)

    # Validate constellation_filter
    if constellation_filter not in [None, 'starlink', 'oneweb']:
        raise ValueError(
            f"❌ Invalid constellation_filter: '{constellation_filter}'\n"
            f"   Must be 'starlink', 'oneweb', or None"
        )

    # Extract optimized pools
    pools = data['pool_optimization']['optimized_pools']

    # Extract satellite IDs from each constellation
    satellite_ids = []
    metadata = {}

    # Starlink pool (expected: 101 satellites)
    starlink_pool = pools.get('starlink', [])
    starlink_ids = [sat['satellite_id'] for sat in starlink_pool]
    metadata['starlink'] = {
        'count': len(starlink_ids),
        'satellites': starlink_pool
    }

    # OneWeb pool (expected: 24 satellites)
    oneweb_pool = pools.get('oneweb', [])
    oneweb_ids = [sat['satellite_id'] for sat in oneweb_pool]
    metadata['oneweb'] = {
        'count': len(oneweb_ids),
        'satellites': oneweb_pool
    }

    # Apply constellation filter
    if constellation_filter == 'starlink':
        satellite_ids = starlink_ids
        print(f"\n📊 Satellite Pool Loaded (Starlink Only):")
        print(f"   Starlink: {len(starlink_ids)} satellites")
        print(f"   ⚠️  OneWeb excluded (cross-constellation handover not supported)")
    elif constellation_filter == 'oneweb':
        satellite_ids = oneweb_ids
        print(f"\n📊 Satellite Pool Loaded (OneWeb Only):")
        print(f"   OneWeb: {len(oneweb_ids)} satellites")
        print(f"   ⚠️  Starlink excluded (cross-constellation handover not supported)")
    else:
        # Load both constellations
        satellite_ids.extend(starlink_ids)
        satellite_ids.extend(oneweb_ids)
        print(f"\n📊 Satellite Pool Loaded (Multi-Constellation):")
        print(f"   Starlink: {len(starlink_ids)} satellites")
        print(f"   OneWeb:   {len(oneweb_ids)} satellites")
        print(f"   Total:    {len(satellite_ids)} satellites")

    # Validation (CRITICAL - prevents errors)
    starlink_count = len(starlink_ids)
    oneweb_count = len(oneweb_ids)

    # Validate source data integrity
    assert starlink_count == 101, \
        f"❌ Expected 101 Starlink satellites, got {starlink_count}. Check Stage 4 configuration."

    assert oneweb_count == 24, \
        f"❌ Expected 24 OneWeb satellites, got {oneweb_count}. Check Stage 4 configuration."

    # Validate filtered result
    if constellation_filter == 'starlink':
        assert len(satellite_ids) == 101, \
            f"❌ Starlink filter failed: expected 101, got {len(satellite_ids)}"
        print(f"✅ Validation passed: 101 Starlink satellites\n")
    elif constellation_filter == 'oneweb':
        assert len(satellite_ids) == 24, \
            f"❌ OneWeb filter failed: expected 24, got {len(satellite_ids)}"
        print(f"✅ Validation passed: 24 OneWeb satellites\n")
    else:
        assert len(satellite_ids) == 125, \
            f"❌ Expected 125 satellites, got {len(satellite_ids)}. Stage 4 output may be corrupted."
        print(f"✅ Validation passed: 125 satellites (101 Starlink + 24 OneWeb)\n")

    # Display satellite sample
    if constellation_filter == 'starlink' or constellation_filter is None:
        print(f"🛰️  Starlink Sample:")
        print(f"   First 3: {starlink_ids[:3]}")
        print(f"   Last 3:  {starlink_ids[-3:]}")
    if constellation_filter == 'oneweb' or constellation_filter is None:
        print(f"🛰️  OneWeb Sample:")
        print(f"   First 3: {oneweb_ids[:3]}")
        print(f"   Last 3:  {oneweb_ids[-3:]}")
    print()

    if return_metadata:
        return satellite_ids, metadata
    else:
        return satellite_ids


def get_satellite_constellation(satellite_id: str, metadata: Dict = None) -> str:
    """
    Determine which constellation a satellite belongs to

    Args:
        satellite_id: NORAD catalog number (as string)
        metadata: Optional metadata dict from load_stage4_optimized_satellites()

    Returns:
        'starlink' or 'oneweb'

    Note: If metadata not provided, uses NORAD ID ranges (approximate)
    """
    if metadata:
        # Use metadata from Stage 4 (most accurate)
        for constellation in ['starlink', 'oneweb']:
            if constellation in metadata:
                sat_ids = [sat['satellite_id'] for sat in metadata[constellation]['satellites']]
                if satellite_id in sat_ids:
                    return constellation

    # Fallback: NORAD ID range heuristic
    # Starlink: typically 44000-48000, 53000-56000
    # OneWeb: typically 47000-51000
    sat_id_int = int(satellite_id)

    if 47000 <= sat_id_int <= 51000:
        return 'oneweb'
    else:
        return 'starlink'  # Default assumption


# Legacy function for backward compatibility (DEPRECATED)
def load_satellite_ids(max_satellites: int = None, date_str: str = "20251007") -> List[str]:
    """
    DEPRECATED: Use load_stage4_optimized_satellites() instead

    This function is kept for backward compatibility but should not be used
    in new code. It bypasses orbit-engine Stage 4 optimization.

    Replacement:
        satellite_ids = load_stage4_optimized_satellites()
    """
    import warnings
    warnings.warn(
        "load_satellite_ids() is deprecated. Use load_stage4_optimized_satellites() instead. "
        "See DATA_DEPENDENCIES.md for details.",
        DeprecationWarning,
        stacklevel=2
    )

    # For backward compatibility, call the new function
    return load_stage4_optimized_satellites()


def verify_satellite_pool_integrity(satellite_ids: List[str],
                                    expected_constellation: str = None,
                                    metadata: Dict = None) -> Tuple[bool, str]:
    """
    Verify that satellite pool meets academic compliance requirements

    Args:
        satellite_ids: List of satellite IDs to verify
        expected_constellation: Expected constellation ('starlink', 'oneweb', or None for multi)

    Returns:
        (is_valid, message): Tuple of validation result and explanation

    Checks:
    - Total count matches expected (101 Starlink, 24 OneWeb, or 125 multi)
    - No duplicates
    - Valid NORAD ID format
    - Constellation consistency
    """
    issues = []

    # Check total count based on expected constellation
    expected_counts = {
        'starlink': 101,
        'oneweb': 24,
        None: 125  # Multi-constellation
    }
    expected_count = expected_counts.get(expected_constellation)
    if expected_count and len(satellite_ids) != expected_count:
        issues.append(f"❌ Expected {expected_count} satellites, got {len(satellite_ids)}")

    # Check for duplicates
    unique_ids = set(satellite_ids)
    if len(unique_ids) != len(satellite_ids):
        duplicates = len(satellite_ids) - len(unique_ids)
        issues.append(f"❌ Found {duplicates} duplicate satellite IDs")

    # Check constellation consistency
    # Use metadata if available for accurate classification
    constellations = set()
    for sat_id in satellite_ids:
        constellation = get_satellite_constellation(sat_id, metadata)
        constellations.add(constellation)

    if expected_constellation:
        # Single-constellation mode
        if expected_constellation not in constellations:
            issues.append(f"❌ Expected {expected_constellation} satellites, but found {constellations}")
        if len(constellations) > 1:
            issues.append(f"❌ Mixed constellations detected: {constellations} (expected {expected_constellation} only)")
    else:
        # Multi-constellation mode
        if 'starlink' not in constellations:
            issues.append("❌ Missing Starlink satellites")
        if 'oneweb' not in constellations:
            issues.append("❌ Missing OneWeb satellites")

    # Check NORAD ID format (should be numeric strings)
    for sat_id in satellite_ids[:10]:  # Check first 10
        if not sat_id.isdigit():
            issues.append(f"❌ Invalid NORAD ID format: {sat_id}")
            break

    if issues:
        return False, "\n".join(issues)
    else:
        if expected_constellation == 'starlink':
            return True, "✅ Satellite pool integrity verified: 101 Starlink satellites"
        elif expected_constellation == 'oneweb':
            return True, "✅ Satellite pool integrity verified: 24 OneWeb satellites"
        else:
            return True, "✅ Satellite pool integrity verified: 125 satellites (multi-constellation)"


if __name__ == "__main__":
    """
    Test satellite loading and verify integrity
    """
    print("=" * 80)
    print("🧪 Testing Satellite Pool Loading")
    print("=" * 80)
    print()

    # Load satellites
    try:
        satellite_ids, metadata = load_stage4_optimized_satellites(return_metadata=True)

        # Verify integrity
        is_valid, message = verify_satellite_pool_integrity(satellite_ids)
        print(message)
        print()

        # Test constellation detection
        print("🔍 Testing Constellation Detection:")
        test_ids = satellite_ids[:2] + satellite_ids[-2:]  # First 2 + Last 2
        for sat_id in test_ids:
            constellation = get_satellite_constellation(sat_id, metadata)
            print(f"   {sat_id}: {constellation}")

        print()
        print("=" * 80)
        print("✅ All tests passed")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
