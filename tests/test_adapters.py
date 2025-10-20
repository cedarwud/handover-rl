#!/usr/bin/env python3
"""
Unit Tests for Orbit-Engine Adapters

Tests:
- TLELoader: TLE file loading and management
- OrbitEngineAdapter: State calculation
"""

import sys
from pathlib import Path
import unittest
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adapters import TLELoader, TLE, OrbitEngineAdapter


class TestTLELoader(unittest.TestCase):
    """Test TLELoader functionality"""

    def setUp(self):
        """Set up test TLE loader"""
        # Note: Update path to actual TLE directory
        self.tle_dir = "../orbit-engine/data/tle_data/starlink/tle"

        # Skip if TLE directory doesn't exist
        if not Path(self.tle_dir).exists():
            self.skipTest(f"TLE directory not found: {self.tle_dir}")

    def test_loader_initialization(self):
        """Test TLELoader initialization"""
        loader = TLELoader(self.tle_dir, "starlink_*.tle")
        count = loader.load_all_tles()

        self.assertGreater(count, 0, "Should load at least one TLE file")
        self.assertGreater(len(loader.get_available_satellites()), 0,
                          "Should have available satellites")

    def test_tle_epoch_parsing(self):
        """Test TLE epoch date parsing"""
        # Example TLE line 1 (epoch: 2025-10-15)
        line1 = "1 55490U 23001A   25288.12345678  .00001234  00000-0  12345-4 0  9991"
        line2 = "2 55490  53.0000 100.0000 0001000  90.0000 270.0000 15.12345678123456"

        tle = TLE("55490", "STARLINK-TEST", line1, line2)

        # Check epoch year and day
        self.assertEqual(tle.epoch.year, 2025)
        # Day 288 of 2025 = October 15
        self.assertEqual(tle.epoch.month, 10)

    def test_tle_validity_check(self):
        """Test TLE validity for target date"""
        line1 = "1 55490U 23001A   25288.12345678  .00001234  00000-0  12345-4 0  9991"
        line2 = "2 55490  53.0000 100.0000 0001000  90.0000 270.0000 15.12345678123456"
        tle = TLE("55490", "STARLINK-TEST", line1, line2)

        # Should be valid within ±2 days from epoch
        target1 = tle.epoch + timedelta(days=1)
        self.assertTrue(tle.is_valid_for_date(target1, max_days=2))

        # Should be invalid beyond ±2 days
        target2 = tle.epoch + timedelta(days=5)
        self.assertFalse(tle.is_valid_for_date(target2, max_days=2))

    def test_get_tle_for_date(self):
        """Test getting TLE for specific date"""
        loader = TLELoader(self.tle_dir, "starlink_*.tle")
        loader.load_all_tles()

        satellites = loader.get_available_satellites()
        if len(satellites) == 0:
            self.skipTest("No satellites loaded")

        sat_id = satellites[0]

        # Get epoch range
        epoch_range = loader.get_epoch_range(sat_id)
        if epoch_range is None:
            self.skipTest(f"No TLEs for satellite {sat_id}")

        # Try to get TLE for a date in the middle of epoch range
        target_date = epoch_range[0] + (epoch_range[1] - epoch_range[0]) / 2
        tle = loader.get_tle_for_date(sat_id, target_date)

        self.assertIsNotNone(tle, "Should find valid TLE")
        self.assertEqual(tle.satellite_id, sat_id)

    def test_tle_sequence_for_period(self):
        """Test TLE sequence generation for 30-day period"""
        loader = TLELoader(self.tle_dir, "starlink_*.tle")
        loader.load_all_tles()

        satellites = loader.get_available_satellites()
        if len(satellites) == 0:
            self.skipTest("No satellites loaded")

        sat_id = satellites[0]

        # Get epoch range
        epoch_range = loader.get_epoch_range(sat_id)
        if epoch_range is None:
            self.skipTest(f"No TLEs for satellite {sat_id}")

        # Generate sequence for first 30 days
        start_date = epoch_range[0]
        end_date = start_date + timedelta(days=30)

        segments = loader.get_tle_sequence_for_period(sat_id, start_date, end_date)

        # Should have segments (may not be full 30 days if TLE coverage is incomplete)
        self.assertGreater(len(segments), 0, "Should have at least one segment")

        # Each segment should have (start, end, tle)
        for start, end, tle in segments:
            self.assertIsInstance(start, datetime)
            self.assertIsInstance(end, datetime)
            self.assertIsInstance(tle, TLE)
            self.assertLess(start, end)


class TestOrbitEngineAdapter(unittest.TestCase):
    """Test OrbitEngineAdapter functionality"""

    def setUp(self):
        """Set up test adapter"""
        self.config = {
            'ground_station': {
                'latitude': 24.9441,
                'longitude': 121.3714,
                'altitude_m': 36.0,
                'min_elevation_deg': 10.0
            },
            'physics': {
                'frequency_ghz': 12.5,
                'bandwidth_mhz': 100,
                'tx_power_dbm': 33.0,
                'use_atmospheric_loss': True
            },
            'data_generation': {
                'tle_strategy': {
                    'tle_directory': '../orbit-engine/data/tle_data/starlink/tle',
                    'file_pattern': 'starlink_*.tle'
                }
            }
        }

        # Skip if TLE directory doesn't exist
        tle_dir = Path(self.config['data_generation']['tle_strategy']['tle_directory'])
        if not tle_dir.exists():
            self.skipTest(f"TLE directory not found: {tle_dir}")

    def test_adapter_initialization(self):
        """Test adapter initialization"""
        try:
            adapter = OrbitEngineAdapter(self.config)
            self.assertIsNotNone(adapter)
            self.assertGreater(len(adapter.tle_loader.get_available_satellites()), 0)
        except ImportError as e:
            self.skipTest(f"Orbit-engine modules not available: {e}")

    def test_state_calculation(self):
        """Test state calculation for a satellite"""
        try:
            adapter = OrbitEngineAdapter(self.config)
        except ImportError:
            self.skipTest("Orbit-engine modules not available")

        satellites = adapter.tle_loader.get_available_satellites()
        if len(satellites) == 0:
            self.skipTest("No satellites available")

        sat_id = satellites[0]

        # Get a valid timestamp
        epoch_range = adapter.tle_loader.get_epoch_range(sat_id)
        if epoch_range is None:
            self.skipTest(f"No TLEs for satellite {sat_id}")

        timestamp = epoch_range[0] + timedelta(hours=12)

        # Calculate state
        state = adapter.calculate_state(sat_id, timestamp)

        # Verify state structure
        self.assertIn('rsrp_dbm', state)
        self.assertIn('rsrq_db', state)
        self.assertIn('rs_sinr_db', state)
        self.assertIn('distance_km', state)
        self.assertIn('elevation_deg', state)
        self.assertIn('doppler_shift_hz', state)
        self.assertIn('radial_velocity_ms', state)
        self.assertIn('atmospheric_loss_db', state)
        self.assertIn('path_loss_db', state)
        self.assertIn('propagation_delay_ms', state)
        self.assertIn('offset_mo_db', state)
        self.assertIn('cell_offset_db', state)

        # Verify state value ranges
        self.assertLessEqual(state['rsrp_dbm'], -30)  # Upper bound
        self.assertGreaterEqual(state['rsrp_dbm'], -156)  # Lower bound
        self.assertGreaterEqual(state['distance_km'], 0)
        self.assertGreaterEqual(state['elevation_deg'], -90)
        self.assertLessEqual(state['elevation_deg'], 90)

        print(f"\n✅ State calculation test passed for satellite {sat_id}")
        print(f"   RSRP: {state['rsrp_dbm']:.2f} dBm")
        print(f"   Distance: {state['distance_km']:.2f} km")
        print(f"   Elevation: {state['elevation_deg']:.2f}°")


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestTLELoader))
    suite.addTests(loader.loadTestsFromTestCase(TestOrbitEngineAdapter))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
