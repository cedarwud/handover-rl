#!/usr/bin/env python3
"""
Quick test to verify Minimum Dwell Time Constraint is working
"""
import sys
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.adapter_wrapper import AdapterWrapper
from src.environments import SatelliteHandoverEnv

def test_dwell_time():
    """Test that dwell time constraint prevents rapid handovers"""
    print("="*70)
    print("DWELL TIME CONSTRAINT TEST")
    print("="*70)

    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup adapter
    adapter = AdapterWrapper(config)

    # Load satellite IDs
    sat_ids_file = Path(__file__).parent.parent / 'data' / 'satellite_ids_from_precompute.txt'
    with open(sat_ids_file, 'r') as f:
        satellite_ids = [line.strip() for line in f if line.strip()]

    # Create environment
    env = SatelliteHandoverEnv(adapter, satellite_ids, config)

    print(f"\n✅ Environment created with:")
    print(f"   Dwell time: {env.min_dwell_time_seconds}s ({env.min_dwell_time_steps} steps)")
    print(f"   Time step: {env.time_step_seconds}s")
    print(f"   → Cannot handover for {env.min_dwell_time_steps} steps after each handover\n")

    # Run test episode
    obs, info = env.reset(options={'start_time': datetime(2025, 10, 26, 12, 0, 0)})

    print(f"Episode started:")
    print(f"   Candidates: {info['num_visible']}/{info['num_candidates']}")
    print(f"   Initial satellite: {info['current_satellite']}\n")

    handovers = []
    blocked_attempts = 0

    # Try to force handover every step
    for step in range(20):
        # Try to switch to a different satellite
        action_mask = env._get_action_mask()
        available_actions = [i for i in range(len(action_mask)) if action_mask[i]]

        if len(available_actions) > 1:
            # Try to switch (action != 0)
            action = available_actions[1] if available_actions[1] != 0 else available_actions[0]
        else:
            action = 0  # Stay

        prev_sat = env.current_satellite
        prev_steps_since_ho = env.steps_since_last_handover

        obs, reward, terminated, truncated, info = env.step(action)

        if info['handover_occurred']:
            handovers.append(step)
            print(f"Step {step:2d}: ✅ HANDOVER {prev_sat} → {info['current_satellite']} "
                  f"(Steps since last HO: {prev_steps_since_ho})")
        else:
            if action != 0 and prev_sat == env.current_satellite:
                blocked_attempts += 1
                if env.steps_since_last_handover < env.min_dwell_time_steps:
                    print(f"Step {step:2d}: 🚫 BLOCKED by dwell time "
                          f"({env.steps_since_last_handover}/{env.min_dwell_time_steps} steps)")

        if terminated or truncated:
            break

    print(f"\n{'='*70}")
    print(f"TEST RESULTS")
    print(f"{'='*70}")
    print(f"Total steps: {step + 1}")
    print(f"Handovers: {len(handovers)}")
    print(f"Blocked attempts: {blocked_attempts}")
    print(f"Handover steps: {handovers}")

    if len(handovers) >= 2:
        gaps = [handovers[i+1] - handovers[i] for i in range(len(handovers)-1)]
        print(f"Gaps between handovers: {gaps} steps")
        print(f"Min gap: {min(gaps)} steps (should be >= {env.min_dwell_time_steps})")

        if all(gap >= env.min_dwell_time_steps for gap in gaps):
            print(f"\n✅ SUCCESS: All handover gaps >= {env.min_dwell_time_steps} steps!")
            print(f"   Dwell time constraint is working correctly.")
        else:
            print(f"\n❌ FAIL: Some gaps < {env.min_dwell_time_steps} steps!")
    else:
        print(f"\n⚠️  Less than 2 handovers, cannot verify gaps")

    # Calculate theoretical max handovers for 600s episode
    episode_steps = int(10 * 60 / env.time_step_seconds)  # 600s / 5s = 120 steps
    theoretical_max = episode_steps // env.min_dwell_time_steps
    print(f"\n📐 Theoretical Analysis:")
    print(f"   Episode duration: {10 * 60}s = {episode_steps} steps")
    print(f"   Dwell time: {env.min_dwell_time_seconds}s = {env.min_dwell_time_steps} steps")
    print(f"   Theoretical max handovers: {episode_steps} / {env.min_dwell_time_steps} = {theoretical_max}")
    print(f"   Expected actual: ~{int(theoretical_max * 0.7)}-{theoretical_max} handovers")
    print(f"   → vs v9_aggressive (21.0 handovers) - significant reduction expected!")

if __name__ == '__main__':
    test_dwell_time()
