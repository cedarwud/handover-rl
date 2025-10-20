#!/usr/bin/env python3
"""
Phase 1 - Step 1.5: Test Multi-Satellite Environment

Verify that the new environment works correctly
"""

import sys
import yaml
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from adapters.orbit_engine_adapter import OrbitEngineAdapter
from environments.satellite_handover_env import SatelliteHandoverEnv
from utils.satellite_utils import load_satellite_ids


def main():
    print("=" * 80)
    print("Phase 1 - Step 1.5: Test Multi-Satellite Environment")
    print("=" * 80)

    # Load config
    print("\n[1/5] Loading configuration...")
    config_path = Path(__file__).parent / 'config' / 'data_gen_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Add environment config if not present
    if 'environment' not in config:
        config['environment'] = {
            'time_step_seconds': 5,
            'episode_duration_minutes': 95,
            'max_visible_satellites': 10,
            'reward': {
                'qos_weight': 1.0,
                'handover_penalty': -0.1,
                'ping_pong_penalty': -0.2,
            }
        }

    print("✅ Config loaded")

    # Initialize adapter
    print("\n[2/5] Initializing OrbitEngineAdapter...")
    adapter = OrbitEngineAdapter(config)
    print("✅ Adapter initialized")

    # Satellite IDs - NO HARDCODING, extract from TLE
    # SOURCE: Space-Track.org TLE data
    print("   Extracting satellites from TLE file...")
    satellite_ids = load_satellite_ids(max_satellites=125)
    print(f"   ✅ Loaded {len(satellite_ids)} satellites from TLE")

    # Create environment
    print("\n[3/5] Creating SatelliteHandoverEnv...")
    env = SatelliteHandoverEnv(adapter, satellite_ids, config)
    print("✅ Environment created")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")

    # Test reset
    print("\n[4/5] Testing environment reset...")
    # Use a time when satellites are likely visible (from our earlier tests)
    start_time = datetime(2025, 10, 7, 12, 0, 0)
    obs, info = env.reset(seed=42, options={'start_time': start_time})
    print("✅ Environment reset successful")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Number of visible satellites: {info['num_visible']}")
    print(f"   Current satellite: {info['current_satellite']}")
    print(f"   Episode start: {info['episode_start']}")

    # Test step
    print("\n[5/5] Testing environment step...")
    total_steps = 20
    episode_rewards = []

    print(f"Running {total_steps} steps...")
    print("-" * 80)

    for step_num in range(total_steps):
        # Take action (for now, just use action 0 = stay)
        action = 0

        # Execute step
        next_obs, reward, terminated, truncated, info = env.step(action)

        episode_rewards.append(reward)

        # Log step info
        if step_num % 5 == 0 or terminated or truncated:
            print(f"Step {step_num + 1:3d}: "
                  f"visible={info['num_visible']:2d}, "
                  f"reward={reward:+.3f}, "
                  f"handovers={info['num_handovers']:2d}, "
                  f"sat={info['current_satellite']}")

        # Check if done
        if terminated or truncated:
            reason = "terminated" if terminated else "truncated"
            print(f"\n   Episode ended at step {step_num + 1} ({reason})")
            break

    # Summary
    print("-" * 80)
    print("\n📊 Test Summary")
    print("-" * 80)

    if len(episode_rewards) > 0:
        print(f"Total steps executed: {len(episode_rewards)}")
        print(f"Total reward: {sum(episode_rewards):.3f}")
        print(f"Average reward: {sum(episode_rewards) / len(episode_rewards):.3f}")
        print(f"Handovers: {info['num_handovers']}")
        print(f"Ping-pongs: {info['num_ping_pongs']}")
        print(f"Average RSRP: {info['avg_rsrp']:.1f} dBm")

    # Verification checks
    print("\n🎓 Academic Compliance Checks")
    print("-" * 80)

    checks_passed = 0
    checks_total = 0

    # Check 1: Multi-satellite state
    checks_total += 1
    if obs.shape == (10, 12):
        print("✅ Multi-satellite state: (10, 12) shape correct")
        checks_passed += 1
    else:
        print(f"❌ Multi-satellite state: Expected (10, 12), got {obs.shape}")

    # Check 2: Dynamic action space
    checks_total += 1
    if env.action_space.n == 11:  # 10 visible + 1 stay
        print("✅ Dynamic action space: 11 actions (stay + 10 satellites)")
        checks_passed += 1
    else:
        print(f"❌ Dynamic action space: Expected 11, got {env.action_space.n}")

    # Check 3: Real physics (no zero states)
    checks_total += 1
    non_zero_rows = (obs != 0).any(axis=1).sum()
    if non_zero_rows > 0:
        print(f"✅ Real physics: {non_zero_rows} satellites with non-zero states")
        checks_passed += 1
    else:
        print("⚠️  Real physics: All states are zero (no visible satellites)")

    # Check 4: Reward function working
    checks_total += 1
    if len(episode_rewards) > 0 and any(r != 0 for r in episode_rewards):
        print("✅ Reward function: Non-zero rewards observed")
        checks_passed += 1
    elif len(episode_rewards) > 0:
        print("⚠️  Reward function: All rewards are zero")
    else:
        print("❌ Reward function: No rewards observed")

    print("\n" + "=" * 80)
    print(f"📊 Compliance Score: {checks_passed}/{checks_total} checks passed")

    if checks_passed == checks_total:
        print("\n✅ ENVIRONMENT VERIFICATION PASSED")
        print("\n🚀 Ready for Phase 2 (DQN Agent Implementation)")
        return True
    else:
        print("\n⚠️  ENVIRONMENT VERIFICATION WARNING")
        print("   Some checks did not pass - review implementation")
        return False


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    success = main()
    sys.exit(0 if success else 1)
