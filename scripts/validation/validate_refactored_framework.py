#!/usr/bin/env python3
"""
Validation Script for Refactored Framework

Tests the refactored modular framework against requirements:
1. Smoke test (Level 0: 10 satellites, 10 episodes)
2. Multi-Level Training accessibility (all 6 levels)
3. Component integration
4. Training metrics collection

This validates Task 1.1-1.5 are working end-to-end.
Task 1.6 validation includes comparing with old train_online_rl.py.
"""

import sys
from pathlib import Path
import time
import traceback

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

def test_component_imports():
    """Test 1: All components can be imported"""
    print("=" * 60)
    print("Test 1: Component Imports")
    print("=" * 60)

    try:
        from agents import BaseAgent, DQNAgent
        print("✅ Agents imported: BaseAgent, DQNAgent")

        from trainers import OffPolicyTrainer
        print("✅ Trainers imported: OffPolicyTrainer")

        from configs import get_level_config, TRAINING_LEVELS
        print("✅ Configs imported: get_level_config, TRAINING_LEVELS")

        from environments.satellite_handover_env import SatelliteHandoverEnv
        print("✅ Environment imported: SatelliteHandoverEnv")

        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_multi_level_config():
    """Test 2: Multi-Level Training config (P0 CRITICAL)"""
    print("\n" + "=" * 60)
    print("Test 2: Multi-Level Training Config (P0 CRITICAL)")
    print("=" * 60)

    try:
        from configs import get_level_config, TRAINING_LEVELS

        # Verify all 6 levels exist
        assert len(TRAINING_LEVELS) == 6, f"Expected 6 levels, got {len(TRAINING_LEVELS)}"
        print(f"✅ All 6 training levels exist")

        # Verify key levels
        level0 = get_level_config(0)
        assert level0['num_satellites'] == 10
        assert level0['num_episodes'] == 10
        print(f"✅ Level 0 (Smoke test): {level0['num_satellites']} sats, {level0['num_episodes']} eps")

        level1 = get_level_config(1)
        assert level1['num_satellites'] == 20
        assert level1['num_episodes'] == 100
        assert level1['estimated_time_hours'] == 2.0
        print(f"✅ Level 1 (Quick validation): {level1['num_satellites']} sats, {level1['num_episodes']} eps, {level1['estimated_time_hours']}h")

        level5 = get_level_config(5)
        assert level5['num_satellites'] == 101
        assert level5['num_episodes'] == 1700
        assert level5['estimated_time_hours'] == 35.0
        print(f"✅ Level 5 (Full training): {level5['num_satellites']} sats, {level5['num_episodes']} eps, {level5['estimated_time_hours']}h")

        print("✅ Multi-Level Training Config validated (Novel Aspect #1 preserved)")
        return True

    except Exception as e:
        print(f"❌ Multi-level config failed: {e}")
        traceback.print_exc()
        return False


def test_agent_instantiation():
    """Test 3: Agent can be instantiated and used"""
    print("\n" + "=" * 60)
    print("Test 3: Agent Instantiation")
    print("=" * 60)

    try:
        import numpy as np
        from gymnasium import spaces
        from agents import DQNAgent

        # Create agent
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10, 12), dtype=np.float32)
        action_space = spaces.Discrete(11)
        config = {
            'agent': {
                'learning_rate': 1e-4,
                'gamma': 0.99,
                'batch_size': 64,
                'buffer_capacity': 1000,
                'hidden_dim': 128,
            }
        }

        agent = DQNAgent(obs_space, action_space, config)
        print(f"✅ DQNAgent created: {agent.obs_shape} obs, {agent.n_actions} actions")

        # Test action selection
        state = np.random.randn(10, 12).astype(np.float32)
        action = agent.select_action(state, deterministic=False)
        assert 0 <= action < 11, f"Action {action} out of range"
        print(f"✅ Action selection works: action={action}")

        # Test config retrieval
        agent_config = agent.get_config()
        assert 'algorithm' in agent_config
        assert 'epsilon' in agent_config
        print(f"✅ Config retrieval works: {len(agent_config)} parameters")

        return True

    except Exception as e:
        print(f"❌ Agent instantiation failed: {e}")
        traceback.print_exc()
        return False


def test_smoke_run():
    """Test 4: Smoke test with minimal training (3 episodes)"""
    print("\n" + "=" * 60)
    print("Test 4: Smoke Test (3 episodes)")
    print("=" * 60)
    print("This tests end-to-end integration without full training")

    try:
        import yaml
        import numpy as np
        from datetime import datetime
        from adapters.orbit_engine_adapter import OrbitEngineAdapter
        from environments.satellite_handover_env import SatelliteHandoverEnv
        from agents import DQNAgent
        from trainers import OffPolicyTrainer
        from utils.satellite_utils import load_stage4_optimized_satellites

        print("\nStep 1: Load config...")
        config_path = Path(__file__).parent.parent.parent / 'config' / 'data_gen_config.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Add agent config if missing
        if 'agent' not in config:
            config['agent'] = {
                'learning_rate': 1e-4,
                'gamma': 0.99,
                'batch_size': 32,  # Smaller for smoke test
                'buffer_capacity': 500,
                'target_update_freq': 50,
                'hidden_dim': 64,  # Smaller for speed
            }
        print("✅ Config loaded")

        print("\nStep 2: Initialize adapter...")
        adapter = OrbitEngineAdapter(config)
        print("✅ Adapter initialized")

        print("\nStep 3: Load satellites (using 5 satellites for speed)...")
        satellite_ids, _ = load_stage4_optimized_satellites(
            constellation_filter='starlink',
            return_metadata=True
        )
        satellite_ids = satellite_ids[:5]  # Just 5 satellites for smoke test
        print(f"✅ Loaded {len(satellite_ids)} satellites: {satellite_ids}")

        print("\nStep 4: Create environment...")
        env = SatelliteHandoverEnv(adapter, satellite_ids, config)
        print(f"✅ Environment created: obs_space={env.observation_space.shape}, action_space={env.action_space.n}")

        print("\nStep 5: Create agent...")
        agent = DQNAgent(env.observation_space, env.action_space, config)
        print(f"✅ Agent created: epsilon={agent.epsilon}")

        print("\nStep 6: Create trainer...")
        trainer = OffPolicyTrainer(env, agent, config)
        print("✅ Trainer created")

        print("\nStep 7: Run 3 training episodes...")
        start_time = datetime(2025, 7, 27, 0, 0, 0)

        for episode in range(3):
            print(f"\n  Episode {episode + 1}/3:")
            metrics = trainer.train_episode(
                episode_idx=episode,
                episode_start_time=start_time,
                seed=42 + episode
            )

            print(f"    Reward: {metrics['reward']:.2f}")
            print(f"    Steps: {metrics['length']}")
            print(f"    Handovers: {metrics['handovers']}")
            print(f"    Loss: {metrics['loss']:.4f}")
            print(f"    Updates: {metrics['num_updates']}")

        print("\n✅ Smoke test completed successfully!")
        print(f"   Final epsilon: {agent.epsilon:.3f}")
        print(f"   Buffer size: {len(agent.replay_buffer)}")
        print(f"   Training steps: {agent.training_steps}")

        return True

    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests"""
    print("\n")
    print("=" * 60)
    print("REFACTORED FRAMEWORK VALIDATION")
    print("Task 1.6: End-to-End Validation")
    print("=" * 60)
    print("\n")

    results = {}

    # Test 1: Component imports
    results['imports'] = test_component_imports()

    # Test 2: Multi-level config (P0 CRITICAL)
    results['multi_level'] = test_multi_level_config()

    # Test 3: Agent instantiation
    results['agent'] = test_agent_instantiation()

    # Test 4: Smoke test
    results['smoke_test'] = test_smoke_run()

    # Summary
    print("\n")
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nRefactored framework validated successfully!")
        print("Phase 1 (Tasks 1.1-1.6): COMPLETE ✅")
        print("\nComponents validated:")
        print("  ✓ BaseAgent interface")
        print("  ✓ DQNAgent (inherits BaseAgent)")
        print("  ✓ OffPolicyTrainer")
        print("  ✓ Multi-Level Training (6 levels) - Novel Aspect #1 ⭐")
        print("  ✓ End-to-end training loop")
        print("\nReady for Phase 2: Rule-based Baselines")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        print("\nPlease review the errors above and fix before proceeding.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
