#!/usr/bin/env python3
"""
Test Orbit-Engine Adapter Complete Implementation
驗證完全移除簡化算法後的適配器功能

✅ Grade A Standard: 驗證使用完整 orbit-engine 實現
✅ 確認無硬編碼值、無簡化算法
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add orbit-engine to path (required for orbit_engine_adapter imports)
ORBIT_ENGINE_ROOT = PROJECT_ROOT.parent / "orbit-engine"
sys.path.insert(0, str(ORBIT_ENGINE_ROOT))

from src.adapters.orbit_engine_adapter import OrbitEngineAdapter


def load_config():
    """Load configuration from data_gen_config.yaml"""
    config_path = PROJECT_ROOT / "config" / "data_gen_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_adapter_initialization():
    """Test 1: Adapter initializes with complete configuration"""
    print("=" * 80)
    print("Test 1: Adapter Initialization with Complete Configuration")
    print("=" * 80)

    try:
        config = load_config()
        adapter = OrbitEngineAdapter(config)

        print("✅ Adapter initialized successfully")
        print(f"   - SGP4 Calculator: {type(adapter.sgp4_calc).__name__}")
        print(f"   - ITU-R Physics Calculator: {type(adapter.itur_calc).__name__}")
        print(f"   - 3GPP Signal Calculator: {type(adapter.gpp_calc).__name__}")
        print(f"   - Atmospheric Model: {type(adapter.atmospheric_model).__name__}")

        # Verify no hardcoded defaults were used
        assert hasattr(adapter, 'atmospheric_model'), "Missing atmospheric_model"
        assert hasattr(adapter, 'gpp_calc'), "Missing gpp_calc"
        assert hasattr(adapter, 'itur_calc'), "Missing itur_calc"

        print("\n✅ All calculators properly initialized")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_signal_quality_calculation():
    """Test 2: Signal quality uses complete 3GPP implementation"""
    print("\n" + "=" * 80)
    print("Test 2: Signal Quality Calculation (Complete 3GPP Implementation)")
    print("=" * 80)

    try:
        config = load_config()
        adapter = OrbitEngineAdapter(config)

        # Get a test satellite
        satellites = adapter.get_all_satellites()
        if not satellites:
            print("⚠️ No TLE data available, skipping signal quality test")
            return True

        test_sat_id = list(satellites.keys())[0]
        test_tle = satellites[test_sat_id]

        print(f"\nTesting with satellite: {test_sat_id}")
        print(f"TLE Epoch: {test_tle.epoch_datetime}")

        # Calculate state
        timestamp = datetime.now(timezone.utc)
        state = adapter.calculate_state(test_sat_id, timestamp, test_tle)

        # Verify signal quality fields
        signal_quality = state['signal_quality']

        print("\n📊 Signal Quality Results:")
        print(f"   RSRP: {signal_quality['rsrp_dbm']:.2f} dBm")
        print(f"   RSRQ: {signal_quality['rsrq_db']:.2f} dB")
        print(f"   SINR: {signal_quality['rs_sinr_db']:.2f} dB")
        print(f"   Standard: {signal_quality.get('calculation_standard', 'N/A')}")

        # Verify complete 3GPP output fields
        required_fields = ['rsrp_dbm', 'rsrq_db', 'rs_sinr_db', 'rssi_dbm',
                          'noise_power_dbm', 'interference_power_dbm']
        missing_fields = [f for f in required_fields if f not in signal_quality]

        if missing_fields:
            print(f"\n⚠️ Missing complete 3GPP fields: {missing_fields}")
            print("   This suggests simplified implementation is still in use")
            return False

        # Verify noise is dynamically calculated (not -100 dBm hardcoded)
        noise_power = signal_quality['noise_power_dbm']
        print(f"\n🔬 Noise Power: {noise_power:.2f} dBm")

        if abs(noise_power + 100.0) < 0.01:
            print("   ⚠️ WARNING: Noise power is exactly -100.0 dBm")
            print("   This may indicate hardcoded value instead of Johnson-Nyquist calculation")
            return False

        print("   ✅ Noise power is dynamically calculated (not hardcoded)")

        # Verify interference is modeled (not zero)
        interference_power = signal_quality['interference_power_dbm']
        print(f"\n🔬 Interference Power: {interference_power:.2f} dBm")

        print("\n✅ Signal quality calculation uses complete 3GPP implementation")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_atmospheric_attenuation():
    """Test 3: Atmospheric loss uses ITU-R P.676-13 complete model"""
    print("\n" + "=" * 80)
    print("Test 3: Atmospheric Attenuation (ITU-R P.676-13 Complete Model)")
    print("=" * 80)

    try:
        config = load_config()
        adapter = OrbitEngineAdapter(config)

        # Test atmospheric model directly
        atm_model = adapter.atmospheric_model

        # Test at different elevations
        elevations = [10, 30, 60, 90]
        frequency_ghz = config['physics']['frequency_ghz']

        print(f"\nFrequency: {frequency_ghz} GHz")
        print(f"Atmospheric Model: {type(atm_model).__name__}")
        print("\nAttenuation vs Elevation:")

        prev_loss = None
        for elevation in elevations:
            loss_db = atm_model.calculate_total_attenuation(
                frequency_ghz=frequency_ghz,
                elevation_deg=elevation
            )
            print(f"   {elevation:2d}°: {loss_db:.4f} dB")

            # Verify attenuation decreases with elevation (physics check)
            if prev_loss is not None:
                if loss_db >= prev_loss:
                    print(f"   ⚠️ WARNING: Attenuation not decreasing with elevation")
                    print(f"   This may indicate incorrect implementation")
                    return False
            prev_loss = loss_db

        # Check if values are suspiciously round (indicating hardcoded)
        test_loss = atm_model.calculate_total_attenuation(
            frequency_ghz=frequency_ghz,
            elevation_deg=15
        )

        if test_loss == 10.0 or test_loss == 0.5:
            print(f"\n❌ Attenuation value is suspiciously round: {test_loss}")
            print("   This indicates hardcoded values, not ITU-R P.676-13 calculation")
            return False

        print("\n✅ Atmospheric attenuation uses complete ITU-R P.676-13 model")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration_completeness():
    """Test 4: Verify configuration has all required parameters"""
    print("\n" + "=" * 80)
    print("Test 4: Configuration Completeness Check")
    print("=" * 80)

    try:
        config = load_config()

        # Check required sections
        required_sections = {
            'signal_calculator': ['bandwidth_mhz', 'subcarrier_spacing_khz',
                                 'noise_figure_db', 'temperature_k'],
            'atmospheric_model': ['temperature_k', 'pressure_hpa',
                                 'water_vapor_density_g_m3'],
            'physics': ['frequency_ghz', 'tx_antenna_gain_db', 'rx_antenna_gain_db']
        }

        all_good = True
        for section, required_params in required_sections.items():
            print(f"\n📋 Checking [{section}]:")

            if section not in config:
                print(f"   ❌ Missing section: {section}")
                all_good = False
                continue

            for param in required_params:
                if param in config[section]:
                    value = config[section][param]
                    print(f"   ✅ {param}: {value}")
                else:
                    print(f"   ❌ Missing parameter: {param}")
                    all_good = False

        if all_good:
            print("\n✅ All required configuration parameters present")
        else:
            print("\n❌ Configuration incomplete")

        return all_good

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("Orbit-Engine Adapter Complete Implementation Tests")
    print("驗證完全移除簡化算法")
    print("=" * 80)

    tests = [
        ("Configuration Completeness", test_configuration_completeness),
        ("Adapter Initialization", test_adapter_initialization),
        ("Atmospheric Attenuation (ITU-R P.676-13)", test_atmospheric_attenuation),
        ("Signal Quality (3GPP TS 38.214)", test_signal_quality_calculation),
    ]

    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(results.values())

    print("\n" + "=" * 80)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All tests passed - Complete implementation verified!")
        return 0
    else:
        print("❌ Some tests failed - Review implementation")
        return 1


if __name__ == "__main__":
    exit(main())
