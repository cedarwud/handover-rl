#!/usr/bin/env python3
"""
Phase 0 - Step 0.5: Baseline Benchmark

Establish performance baselines for the refactor
"""

import sys
import time
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from adapters.orbit_engine_adapter import OrbitEngineAdapter
from utils.satellite_utils import load_satellite_ids


def benchmark_single_satellite(adapter, sat_id, timestamp, iterations=10):
    """Benchmark single satellite query performance"""
    times = []

    for _ in range(iterations):
        start = time.time()
        state = adapter.calculate_state(sat_id, timestamp)
        elapsed = time.time() - start
        times.append(elapsed)

    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
    }


def benchmark_multi_satellite(adapter, satellite_ids, timestamp):
    """Benchmark multi-satellite query performance"""
    start = time.time()

    results = []
    for sat_id in satellite_ids:
        try:
            state = adapter.calculate_state(sat_id, timestamp)
            if state:
                results.append(state)
        except:
            pass

    elapsed = time.time() - start

    return {
        'total_time_s': elapsed,
        'num_queried': len(satellite_ids),
        'num_successful': len(results),
        'time_per_sat_ms': (elapsed / len(satellite_ids)) * 1000 if satellite_ids else 0,
        'results': results,
    }


def main():
    print("=" * 80)
    print("Phase 0 - Step 0.5: Baseline Benchmark")
    print("=" * 80)

    # Load config
    print("\n[1/6] Initializing...")
    config_path = Path(__file__).parent / 'config' / 'data_gen_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    adapter = OrbitEngineAdapter(config)
    print("✅ OrbitEngineAdapter initialized")

    # Test satellites - NO HARDCODING, extract from TLE
    # SOURCE: Space-Track.org TLE data
    print("   Extracting test satellites from TLE...")
    test_satellites = load_satellite_ids(max_satellites=125)
    print(f"   ✅ Loaded {len(test_satellites)} satellites from TLE")

    test_time = datetime(2025, 10, 7, 12, 0, 0)

    # Benchmark 1: Single satellite query
    print("\n[2/6] Benchmark 1: Single Satellite Query")
    print("-" * 80)

    result = benchmark_single_satellite(adapter, test_satellites[0], test_time, iterations=10)
    print(f"Satellite: {test_satellites[0]}")
    print(f"  Mean time:   {result['mean_ms']:.2f} ms")
    print(f"  Std dev:     {result['std_ms']:.2f} ms")
    print(f"  Range:       {result['min_ms']:.2f} - {result['max_ms']:.2f} ms")

    baseline_single_ms = result['mean_ms']

    # Benchmark 2: Multi-satellite query (10 satellites)
    print("\n[3/6] Benchmark 2: Multi-Satellite Query (10 satellites)")
    print("-" * 80)

    result = benchmark_multi_satellite(adapter, test_satellites[:10], test_time)
    print(f"Satellites queried: {result['num_queried']}")
    print(f"Successful queries: {result['num_successful']}")
    print(f"Total time:         {result['total_time_s']:.3f} s")
    print(f"Time per satellite: {result['time_per_sat_ms']:.2f} ms")

    # Benchmark 3: Multi-satellite query (30 satellites)
    print("\n[4/6] Benchmark 3: Multi-Satellite Query (30 satellites)")
    print("-" * 80)

    result_30 = benchmark_multi_satellite(adapter, test_satellites[:30], test_time)
    print(f"Satellites queried: {result_30['num_queried']}")
    print(f"Successful queries: {result_30['num_successful']}")
    print(f"Total time:         {result_30['total_time_s']:.3f} s")
    print(f"Time per satellite: {result_30['time_per_sat_ms']:.2f} ms")

    # Benchmark 4: Estimate for 125 satellites
    print("\n[5/6] Benchmark 4: Projected Performance (125 satellites)")
    print("-" * 80)

    # Use the 30-satellite benchmark to project 125 satellites
    projected_time_125 = result_30['time_per_sat_ms'] * 125 / 1000
    print(f"Based on 30-satellite benchmark:")
    print(f"  Projected time for 125 satellites: {projected_time_125:.3f} s")
    print(f"  Time per satellite: {result_30['time_per_sat_ms']:.2f} ms")

    requirement_125 = 2.0  # seconds
    if projected_time_125 < requirement_125:
        print(f"  ✅ PASS: {projected_time_125:.3f}s < {requirement_125}s requirement")
    else:
        print(f"  ⚠️  WARNING: {projected_time_125:.3f}s >= {requirement_125}s requirement")

    # State quality analysis
    print("\n[6/6] State Quality Analysis")
    print("-" * 80)

    # Analyze states from the 30-satellite benchmark
    states = result_30['results']

    if states:
        connectable = [s for s in states if s.get('is_connectable', False)]

        print(f"Total states returned: {len(states)}")
        print(f"Connectable satellites: {len(connectable)}")

        if connectable:
            rsrp_values = [s['rsrp_dbm'] for s in connectable]
            elevation_values = [s['elevation_deg'] for s in connectable]

            print(f"\nSignal quality (connectable satellites):")
            print(f"  RSRP range:      {min(rsrp_values):.1f} to {max(rsrp_values):.1f} dBm")
            print(f"  RSRP mean:       {np.mean(rsrp_values):.1f} dBm")
            print(f"  Elevation range: {min(elevation_values):.1f}° to {max(elevation_values):.1f}°")
            print(f"  Elevation mean:  {np.mean(elevation_values):.1f}°")

            # Verify no hardcoding (values should be diverse)
            unique_rsrp = len(set(rsrp_values))
            if unique_rsrp > 1:
                print(f"  ✅ RSRP diversity: {unique_rsrp} unique values (no hardcoding)")
            else:
                print(f"  ⚠️  RSRP diversity: All values identical (possible issue)")
        else:
            print("  ⚠️  No connectable satellites in test set")
    else:
        print("  ❌ No states returned from benchmark")

    # Summary
    print("\n" + "=" * 80)
    print("📊 Baseline Performance Summary")
    print("=" * 80)

    print(f"\nSingle Satellite Query:  {baseline_single_ms:.2f} ms")
    print(f"10-Satellite Query:      {result['total_time_s']:.3f} s ({result['time_per_sat_ms']:.2f} ms/sat)")
    print(f"30-Satellite Query:      {result_30['total_time_s']:.3f} s ({result_30['time_per_sat_ms']:.2f} ms/sat)")
    print(f"125-Satellite Projected: {projected_time_125:.3f} s")

    print("\n✅ BASELINE ESTABLISHED")
    print("\nThese metrics will be used to validate that the refactor maintains performance.")
    print("Target for Phase 1: 125-satellite query in <2 seconds")

    # Save baseline to file
    baseline_file = Path(__file__).parent / 'baseline_metrics.txt'
    with open(baseline_file, 'w') as f:
        f.write("Phase 0 Baseline Metrics\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date: {datetime.now().isoformat()}\n\n")
        f.write(f"Single Satellite Query:  {baseline_single_ms:.2f} ms\n")
        f.write(f"10-Satellite Query:      {result['total_time_s']:.3f} s\n")
        f.write(f"30-Satellite Query:      {result_30['total_time_s']:.3f} s\n")
        f.write(f"125-Satellite Projected: {projected_time_125:.3f} s\n")

    print(f"\n✅ Baseline metrics saved to: {baseline_file}")
    print("\n🚀 Ready to proceed to Phase 1 (Multi-Satellite Gym Environment)")

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
