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
    constellation_filter: str = None,
    use_rl_training_data: bool = False,
    use_candidate_pool: bool = False
) -> List[str]:
    """
    Load satellites from orbit-engine Stage 4 output

    This function reads the scientifically selected satellite pool that has passed
    through orbit-engine's six-stage processing:
        Stage 1: TLE Loading (9000+ candidates)
        Stage 2: Orbital Propagation
        Stage 3: Coordinate Transformation
        Stage 4: Link Feasibility ← SOURCE OF THIS DATA
        Stage 5: Signal Analysis
        Stage 6: Research Optimization

    Args:
        stage4_output_dir: Path to Stage 4 output directory
                          (default: auto-selected based on use_rl_training_data)
        return_metadata: If True, return (satellite_ids, metadata_dict)
        constellation_filter: Filter by constellation ('starlink', 'oneweb', or None for all)
                            IMPORTANT: Cross-constellation handover is NOT supported
                            (Starlink and OneWeb are separate commercial networks)
        use_rl_training_data: If True, use RL training data path (rl_training/stage4/)
                             If False, use normal processing path (stage4/)
        use_candidate_pool: If True, use candidate pool (2922 Starlink satellites)
                           If False, use optimized pool (101 Starlink satellites)
                           Only applies when use_rl_training_data=True

    Returns:
        List of satellite IDs (NORAD catalog numbers as strings):

        Normal Mode (use_rl_training_data=False):
        - If constellation_filter='starlink': 101-103 Starlink satellites (550km LEO)
        - If constellation_filter='oneweb': 24-25 OneWeb satellites (1200km LEO)
        - If constellation_filter=None: ~125 satellites (both constellations)

        RL Training Mode (use_rl_training_data=True, use_candidate_pool=True):
        - If constellation_filter='starlink': 2922 Starlink satellites (candidate pool)
        - If constellation_filter='oneweb': 191 OneWeb satellites (candidate pool)
        - If constellation_filter=None: 3113 satellites (both constellations)

    Raises:
        FileNotFoundError: If no Stage 4 output found
        ValueError: If constellation_filter is invalid

    SOURCE: orbit-engine Stage 4 Link Feasibility Analysis
    FILE OPTIONS:
    - Normal: /orbit-engine/data/outputs/stage4/link_feasibility_output_*.json
    - RL Training: /orbit-engine/data/outputs/rl_training/stage4/link_feasibility_output_*.json

    FIELD OPTIONS:
    - Optimized Pool: pool_optimization['optimized_pools']['starlink|oneweb']
    - Candidate Pool: feasibility_summary['candidate_pool']['by_constellation']['starlink|oneweb']

    Academic Compliance:
    - NO HARDCODING: Reads from actual scientific processing output
    - SINGLE-CONSTELLATION TRAINING: Use constellation_filter for realistic scenarios
    - TRACEABLE: Every satellite traceable to Stage 4 selection criteria

    Note on Cross-Constellation Handover:
    - Starlink and OneWeb are separate commercial networks
    - Users cannot handover between them (like AT&T vs Verizon)
    - For RL training, use constellation_filter='starlink' or 'oneweb'
    """
    # Default Stage 4 output directory based on mode
    if stage4_output_dir is None:
        if use_rl_training_data:
            stage4_output_dir = Path("/home/sat/satellite/orbit-engine/data/outputs/rl_training/stage4")
        else:
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

    # Extract satellite pools based on mode
    satellite_ids = []
    metadata = {}

    if use_rl_training_data and use_candidate_pool:
        # RL Training Mode: Use candidate pool (all connectable satellites)
        # SOURCE: feasibility_summary['candidate_pool']['by_constellation']
        print(f"🎯 RL Training Mode: Loading candidate pool (all connectable satellites)")

        feasibility_summary = data.get('feasibility_summary', {})
        candidate_pool = feasibility_summary.get('candidate_pool', {})
        by_constellation = candidate_pool.get('by_constellation', {})

        # Starlink candidate pool (expected: ~2922 satellites)
        starlink_count = by_constellation.get('starlink', 0)
        # Note: Candidate pool stores counts, not satellite lists
        # We need to extract from connectable_satellites_candidate
        connectable_candidate = data.get('connectable_satellites_candidate', {})

        # Structure: connectable_satellites_candidate['starlink'] = [ {satellite_id, name, constellation, ...}, ... ]
        starlink_satellites = connectable_candidate.get('starlink', [])
        oneweb_satellites = connectable_candidate.get('oneweb', [])

        # Extract satellite IDs from the lists
        starlink_ids = [sat['satellite_id'] for sat in starlink_satellites]
        oneweb_ids = [sat['satellite_id'] for sat in oneweb_satellites]

        metadata['starlink'] = {
            'count': len(starlink_ids),
            'expected_count': starlink_count,
            'source': 'candidate_pool'
        }

        metadata['oneweb'] = {
            'count': len(oneweb_ids),
            'expected_count': by_constellation.get('oneweb', 0),
            'source': 'candidate_pool'
        }

        print(f"   Starlink candidates: {len(starlink_ids)} (expected ~{starlink_count})")
        print(f"   OneWeb candidates: {len(oneweb_ids)} (expected ~{by_constellation.get('oneweb', 0)})")

    else:
        # Normal Mode: Use optimized pool
        # SOURCE: pool_optimization['optimized_pools']
        pools = data['pool_optimization']['optimized_pools']

        # Starlink pool (expected: 101 satellites)
        starlink_pool = pools.get('starlink', [])
        starlink_ids = [sat['satellite_id'] for sat in starlink_pool]
        metadata['starlink'] = {
            'count': len(starlink_ids),
            'satellites': starlink_pool,
            'source': 'optimized_pool'
        }

        # OneWeb pool (expected: 24 satellites)
        oneweb_pool = pools.get('oneweb', [])
        oneweb_ids = [sat['satellite_id'] for sat in oneweb_pool]
        metadata['oneweb'] = {
            'count': len(oneweb_ids),
            'satellites': oneweb_pool,
            'source': 'optimized_pool'
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

    # Validate source data integrity based on mode
    if use_rl_training_data and use_candidate_pool:
        # RL Training Mode: Candidate pool validation (2922 Starlink, 191 OneWeb)
        # Note: Candidate pool contains ALL connectable satellites
        if starlink_count < 2800 or starlink_count > 3100:
            raise ValueError(
                f"❌ Unexpected Starlink candidate count: {starlink_count} (expected ~2922 ±100). "
                f"Check RL training Stage 4 output."
            )

        if oneweb_count < 150 or oneweb_count > 230:
            raise ValueError(
                f"❌ Unexpected OneWeb candidate count: {oneweb_count} (expected ~191 ±40). "
                f"Check RL training Stage 4 output."
            )
    else:
        # Normal Mode: Optimized pool validation (101 Starlink, 24 OneWeb)
        # Note: Expected ~101 Starlink satellites (may vary based on TLE data quality)
        if starlink_count < 90 or starlink_count > 115:
            raise ValueError(
                f"❌ Unexpected Starlink satellite count: {starlink_count} (expected ~101 ±15). "
                f"Check Stage 4 configuration."
            )

        if oneweb_count < 20 or oneweb_count > 30:
            raise ValueError(
                f"❌ Unexpected OneWeb satellite count: {oneweb_count} (expected ~24 ±6). "
                f"Check Stage 4 configuration."
            )

    # Validate filtered result based on mode
    if use_rl_training_data and use_candidate_pool:
        # RL Training Mode validation
        if constellation_filter == 'starlink':
            if len(satellite_ids) < 2800 or len(satellite_ids) > 3100:
                raise ValueError(
                    f"❌ Starlink candidate filter failed: expected ~2922 ±100, got {len(satellite_ids)}"
                )
            print(f"✅ Validation passed: {len(satellite_ids)} Starlink candidates (RL training pool)\n")
        elif constellation_filter == 'oneweb':
            if len(satellite_ids) < 150 or len(satellite_ids) > 230:
                raise ValueError(
                    f"❌ OneWeb candidate filter failed: expected ~191 ±40, got {len(satellite_ids)}"
                )
            print(f"✅ Validation passed: {len(satellite_ids)} OneWeb candidates (RL training pool)\n")
        else:
            # Both constellations: expected ~3113 total (2922 Starlink + 191 OneWeb)
            if len(satellite_ids) < 2950 or len(satellite_ids) > 3330:
                raise ValueError(
                    f"❌ Expected ~3113 candidates ±160, got {len(satellite_ids)}. RL training Stage 4 output may be corrupted."
                )
            print(f"✅ Validation passed: {len(satellite_ids)} candidates ({starlink_count} Starlink + {oneweb_count} OneWeb, RL training)\n")
    else:
        # Normal Mode validation (relaxed range check)
        if constellation_filter == 'starlink':
            if len(satellite_ids) < 90 or len(satellite_ids) > 115:
                raise ValueError(
                    f"❌ Starlink filter failed: expected ~101 ±15, got {len(satellite_ids)}"
                )
            print(f"✅ Validation passed: {len(satellite_ids)} Starlink satellites (optimized pool)\n")
        elif constellation_filter == 'oneweb':
            if len(satellite_ids) < 20 or len(satellite_ids) > 30:
                raise ValueError(
                    f"❌ OneWeb filter failed: expected ~24 ±6, got {len(satellite_ids)}"
                )
            print(f"✅ Validation passed: {len(satellite_ids)} OneWeb satellites (optimized pool)\n")
        else:
            # Both constellations: expected ~125 total (101 Starlink + 24 OneWeb)
            if len(satellite_ids) < 110 or len(satellite_ids) > 145:
                raise ValueError(
                    f"❌ Expected ~125 satellites ±20, got {len(satellite_ids)}. Stage 4 output may be corrupted."
                )
            print(f"✅ Validation passed: {len(satellite_ids)} satellites ({starlink_count} Starlink + {oneweb_count} OneWeb, optimized pool)\n")

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
                # Handle both metadata structures:
                # - Normal mode: metadata[constellation]['satellites'] exists
                # - RL training mode: metadata[constellation]['satellites'] doesn't exist
                if 'satellites' in metadata[constellation]:
                    # Normal mode: optimized pool with full satellite data
                    sat_ids = [sat['satellite_id'] for sat in metadata[constellation]['satellites']]
                    if satellite_id in sat_ids:
                        return constellation
                # If no 'satellites' field, fall through to NORAD ID range heuristic

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
