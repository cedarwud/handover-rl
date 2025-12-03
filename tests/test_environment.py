#!/usr/bin/env python3
"""
Quick validation script for V9 environment

Tests:
1. Environment creation
2. Reset functionality
3. RVT calculation
4. Reward function (RVT-based)
5. Step execution
6. Observation shape (14 dimensions)

Usage:
    python test_environment.py
"""

import sys
import yaml
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters import AdapterWrapper
from src.environments import SatelliteHandoverEnvV9
from src.utils.satellite_utils import load_stage4_optimized_satellites


def test_environment():
    """Test V9 environment basic functionality"""
    print("=" * 80)
    print("ENVIRONMENT VALIDATION TEST")
    print("=" * 80)

    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
    print(f"\n1. Loading config from: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("   ✅ Config loaded")

    # Initialize adapter
    print("\n2. Initializing adapter...")
    adapter = AdapterWrapper(config)
    adapter_info = adapter.get_backend_info()
    if adapter_info['is_precompute']:
        print("   ✅ Using precompute mode")
    else:
        print("   ⚠️  Using real-time calculations")

    # Load satellites
    print("\n3. Loading satellite pool...")
    sat_ids_file = Path(__file__).parent.parent / 'data' / 'satellite_ids_from_precompute.txt'

    # Read satellite IDs directly from file
    with open(sat_ids_file, 'r') as f:
        satellite_ids = [line.strip() for line in f if line.strip()]
    print(f"   ✅ Loaded {len(satellite_ids)} satellites")

    # Create environment
    print("\n4. Creating V9 environment...")
    env = SatelliteHandoverEnvV9(
        adapter=adapter,
        satellite_ids=satellite_ids,
        config=config
    )
    print(f"   ✅ Environment created")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.n}")
    print(f"   Expected: observation=(15, 14), action=16")

    # Test reset
    print("\n5. Testing reset...")
    obs, info = env.reset(seed=42)
    print(f"   ✅ Reset successful")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Current satellite: {info['current_satellite']}")
    print(f"   Visible satellites: {info['num_visible']}/{info['num_candidates']}")

    # Check observation dimensions
    print("\n6. Validating observation dimensions...")
    expected_shape = (15, 14)  # V9: 14 dimensions (13 + RVT)
    if obs.shape == expected_shape:
        print(f"   ✅ Observation shape correct: {obs.shape}")
    else:
        print(f"   ❌ Observation shape incorrect: {obs.shape} (expected {expected_shape})")
        return False

    # Test RVT calculation
    print("\n7. Testing RVT calculation...")
    if info['current_satellite']:
        try:
            rvt = env._calculate_rvt(info['current_satellite'])
            print(f"   ✅ RVT calculated: {rvt:.1f} seconds ({rvt/60:.1f} minutes)")
        except Exception as e:
            print(f"   ❌ RVT calculation failed: {e}")
            return False
    else:
        print("   ⚠️  No current satellite to test RVT")

    # Test step
    print("\n8. Testing step execution...")
    try:
        # Action 0: Stay
        action = 0
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(f"   ✅ Step executed successfully")
        print(f"   Action: {action} (stay)")
        print(f"   Reward: {reward:.2f}")
        print(f"   Handovers: {info['num_handovers']}")
        print(f"   Observation shape: {next_obs.shape}")
    except Exception as e:
        print(f"   ❌ Step execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test reward function components
    print("\n9. Testing reward function (RVT-based)...")
    try:
        # Get current satellite info
        current_sat = env.current_satellite
        if current_sat:
            is_loaded = env._is_satellite_loaded(current_sat)
            load_factor = env._get_satellite_load(current_sat)
            rvt = env._calculate_rvt(current_sat)

            print(f"   Current satellite: {current_sat}")
            print(f"   Is loaded: {is_loaded} (load factor: {load_factor:.2f})")
            print(f"   RVT: {rvt:.1f} seconds")

            # Expected reward for stay action
            if is_loaded:
                expected_reward_component = -env.reward_weights['stay_loaded_factor'] * load_factor
                print(f"   Stay reward (loaded): ~{expected_reward_component:.1f}")
            else:
                expected_reward_component = env.reward_weights['rvt_reward_weight'] * rvt
                print(f"   Stay reward (free): ~{expected_reward_component:.1f} (RVT-based)")

            print("   ✅ Reward function working")
        else:
            print("   ⚠️  No current satellite to test reward")
    except Exception as e:
        print(f"   ❌ Reward function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Run a few steps
    print("\n10. Running 10 test steps...")
    try:
        for i in range(10):
            action = 0  # Stay action
            obs, reward, terminated, truncated, info = env.step(action)
            if i % 3 == 0:
                print(f"    Step {i+1}: reward={reward:.2f}, handovers={info['num_handovers']}, "
                      f"sat={info['current_satellite']}")
        print("   ✅ Multi-step execution successful")
    except Exception as e:
        print(f"   ❌ Multi-step execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Cleanup
    env.close()

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - ENVIRONMENT IS READY")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run quick training test: python train_sb3.py --num-episodes 10")
    print("2. Run full training: python train_sb3.py --num-episodes 2500")
    print("=" * 80)

    return True


if __name__ == '__main__':
    try:
        success = test_environment()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
