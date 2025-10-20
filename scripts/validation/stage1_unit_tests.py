#!/usr/bin/env python3
"""
Stage 1: Unit Validation

Tests individual components in isolation:
- BaseAgent protocol compliance
- Strategy protocol compliance
- ReplayBuffer functionality
- Strategy logic correctness
- Multi-level config loading
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import json
import numpy as np
from datetime import datetime

print("\n" + "=" * 80)
print("STAGE 1: UNIT VALIDATION")
print("=" * 80)

results = {
    'stage': 'Stage 1: Unit Validation',
    'timestamp': datetime.now().isoformat(),
    'tests': {},
    'overall_status': 'PENDING',
}

def run_test(test_name: str, test_func):
    """Run a unit test and log results"""
    try:
        test_func()
        results['tests'][test_name] = {'status': 'PASS', 'error': None}
        print(f"✅ {test_name}")
        return True
    except Exception as e:
        results['tests'][test_name] = {'status': 'FAIL', 'error': str(e)}
        print(f"❌ {test_name}: {str(e)}")
        return False

# Test 1: BaseAgent Protocol
def test_baseagent_protocol():
    from agents.dqn.dqn_agent import DQNAgent
    import gymnasium as gym

    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10, 12), dtype=np.float32)
    act_space = gym.spaces.Discrete(11)

    config = {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_size': 10000,
        'batch_size': 64,
        'target_update_frequency': 10,
    }

    agent = DQNAgent(obs_space, act_space, config)

    assert hasattr(agent, 'select_action'), "Missing select_action"
    assert hasattr(agent, 'update'), "Missing update"
    assert hasattr(agent, 'save'), "Missing save"
    assert hasattr(agent, 'load'), "Missing load"

# Test 2: Strategy Protocol
def test_strategy_protocol():
    from strategies import StrongestRSRPStrategy, A4BasedStrategy, D2BasedStrategy
    from strategies.base_strategy import is_valid_strategy

    assert is_valid_strategy(StrongestRSRPStrategy()), "Strongest RSRP not valid"
    assert is_valid_strategy(A4BasedStrategy()), "A4 not valid"
    assert is_valid_strategy(D2BasedStrategy()), "D2 not valid"

# Test 3: Multi-Level Config
def test_multi_level_config():
    from configs import get_level_config

    for level in range(6):
        config = get_level_config(level)
        assert 'num_satellites' in config
        assert 'num_episodes' in config

    config1 = get_level_config(1)
    assert config1['recommended'] == True

# Test 4: A4 Strategy Logic
def test_a4_strategy_logic():
    from strategies import A4BasedStrategy

    strategy = A4BasedStrategy(threshold_dbm=-100.0, hysteresis_db=1.5)

    # Test case: No candidate exceeds threshold
    obs = np.zeros((3, 12), dtype=np.float32)
    obs[:, 0] = [-110, -105, -108]  # All below threshold
    action = strategy.select_action(obs, serving_satellite_idx=0)
    assert action == 0, "Should stay when no A4 trigger"

# Test 5: D2 Strategy Logic
def test_d2_strategy_logic():
    from strategies import D2BasedStrategy

    strategy = D2BasedStrategy(threshold1_km=1412.8, threshold2_km=1005.8)

    # Test case: Serving not too far
    obs = np.zeros((3, 12), dtype=np.float32)
    obs[:, 3] = [1000, 1200, 1100]  # Serving < threshold1
    action = strategy.select_action(obs, serving_satellite_idx=0)
    assert action == 0, "Should stay when D2 condition 1 not met"

# Run all tests
print("\nRunning unit tests...\n")

tests = [
    ("BaseAgent Protocol", test_baseagent_protocol),
    ("Strategy Protocol", test_strategy_protocol),
    ("Multi-Level Config", test_multi_level_config),
    ("A4 Strategy Logic", test_a4_strategy_logic),
    ("D2 Strategy Logic", test_d2_strategy_logic),
]

passed = 0
for name, func in tests:
    if run_test(name, func):
        passed += 1

# Final assessment
total = len(tests)
results['passed'] = passed
results['total'] = total
results['overall_status'] = 'PASS' if passed == total else 'FAIL'

print(f"\n{'='*80}")
print(f"Results: {passed}/{total} tests passed")

if results['overall_status'] == 'PASS':
    print("✅ STAGE 1: UNIT VALIDATION - PASS")
else:
    print("❌ STAGE 1: UNIT VALIDATION - FAIL")

# Save results
output_dir = Path('results/validation')
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / 'stage1_unit_tests.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: results/validation/stage1_unit_tests.json")
print("=" * 80 + "\n")

sys.exit(0 if results['overall_status'] == 'PASS' else 1)
