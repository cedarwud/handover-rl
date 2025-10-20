# Comprehensive Validation Plan for Refactored RL Framework

**Project**: LEO Satellite Handover RL Baseline Framework
**Date**: 2025-10-20
**Purpose**: Establish academically rigorous baseline for future algorithm comparison
**Validation Type**: Full validation (~4-5 hours)

---

## Executive Summary

This validation plan ensures the refactored RL framework meets **academic research standards** and provides a **trustworthy baseline** for future handover optimization algorithms.

**Core Validation Principle**:
> "This framework must be publication-ready with zero tolerance for hardcoded data, mock simulations, or simplified algorithms."

---

## Validation Priority Hierarchy

### P0 Priority: Academic Compliance (CRITICAL - Must Pass 100%)

**Validation Goal**: Ensure framework meets academic publication standards

**Key Requirements**:
- ❌ **NO hardcoded data** - All parameters from configs/standards
- ❌ **NO mock/simulated data** - All data from real sources (TLE, orbit-engine)
- ❌ **NO simplified algorithms** - Full implementations only (3GPP, ITU-R, DQN)
- ✅ **Full parameter traceability** - Every value traceable to source

**Why P0 Priority?**
- This framework will be the **baseline for all future research**
- Any academic compromise undermines all future comparisons
- Reviewers will scrutinize data sources and algorithm fidelity
- "Garbage baseline = garbage conclusions"

---

### P1 Priority: Functional Completeness

**Validation Goal**: Ensure all components work correctly

**Components to Validate**:
1. DQN training pipeline (BaseAgent + OffPolicyTrainer)
2. Multi-Level Training Strategy (Novel Aspect #1)
3. Rule-based baselines (Strongest RSRP, A4, D2)
4. Unified evaluation framework

---

## Validation Stages

### Stage 0: Academic Compliance Validation ⭐ (P0 - CRITICAL)

**Duration**: ~5 minutes
**Script**: `scripts/validation/stage0_academic_compliance.py`

**Validation Items**:

#### 1. Data Source Verification
- [x] Orbital data comes from real TLE (Space-Track.org via orbit-engine)
- [x] RSRP calculation uses ITU-R official models (not simplified)
- [x] Distance calculation uses full SGP4 (not approximations)
- [x] No `np.random()` or `random.normal()` for fake data generation

**How to Verify**:
```python
# Check orbit-engine adapter uses real TLE
assert adapter loads from orbit-engine Stage 4 output
assert no random data generation in adapter

# Check signal calculations use ITU-R
grep -r "np.random" src/  # Should find NONE for data generation
grep -r "simplified" src/  # Should find NONE
grep -r "mock" src/       # Should find NONE
```

#### 2. Algorithm Completeness Verification
- [x] DQN is full implementation (target network, replay buffer, epsilon decay)
- [x] A4 Event follows 3GPP TS 38.331 Section 5.5.4.5 exactly
- [x] D2 Event follows 3GPP TS 38.331 Section 5.5.4.15a (Rel-17 NTN)
- [x] RSRP calculation follows 3GPP TS 38.214

**How to Verify**:
```python
# DQN completeness check
assert DQNAgent has target_network
assert DQNAgent has replay_buffer
assert DQNAgent has epsilon_decay
assert DQNAgent.update() performs target network sync

# A4/D2 standard compliance check
assert A4BasedStrategy logic matches 3GPP TS 38.331 Section 5.5.4.5
assert D2BasedStrategy logic matches 3GPP TS 38.331 Section 5.5.4.15a
```

#### 3. Parameter Traceability Verification
- [x] All D2 thresholds traceable to orbit-engine Stage 4 (71-day TLE analysis)
- [x] All A4 parameters traceable to Yu et al. 2022 + 3GPP standards
- [x] All DQN hyperparameters in config files (not hardcoded)
- [x] All magic numbers eliminated

**How to Verify**:
```bash
# Check all config files have SOURCE annotations
for cfg in config/strategies/*.yaml; do
    grep -q "SOURCE:" $cfg || echo "Missing SOURCE in $cfg"
done

# Check D2 parameters from orbit-engine
grep "1412.8" config/strategies/d2_based.yaml  # threshold1
grep "1005.8" config/strategies/d2_based.yaml  # threshold2
grep "orbit-engine Stage 4" config/strategies/d2_based.yaml  # source

# Check A4 parameters from standards
grep "100.0" config/strategies/a4_based.yaml  # threshold (Yu et al. 2022)
grep "Yu et al. 2022" config/strategies/a4_based.yaml  # source
```

#### 4. Forbidden Content Check
Search codebase for academic red flags:

**Forbidden Keywords** (must NOT exist):
- "simplified algorithm"
- "mock data"
- "fake data"
- "estimated value"
- "assumed parameter"
- "placeholder implementation"

**Allowed Context** (exception handling):
- "simplified" in comments explaining why we DON'T simplify
- "mock" in test file names only

**Validation Script**:
```bash
# Red flag search
grep -rn "simplified" src/ --include="*.py" | grep -v "not simplified"
grep -rn "mock" src/ --include="*.py" | grep -v "test"
grep -rn "fake" src/ --include="*.py"
grep -rn "estimated" src/ --include="*.py" | grep -v "estimated_time"
```

#### 5. Standards Compliance Matrix

| Component | Standard | Section | Status |
|-----------|----------|---------|--------|
| A4 Event | 3GPP TS 38.331 v18.5.1 | 5.5.4.5 | ✅ Verify |
| D2 Event | 3GPP TS 38.331 v18.5.1 | 5.5.4.15a | ✅ Verify |
| RSRP Calc | 3GPP TS 38.214 | - | ✅ Verify |
| Path Loss | ITU-R P.676 | - | ✅ Verify |
| Orbit Mech | SGP4 (NORAD) | - | ✅ Verify |
| TLE Data | Space-Track.org | - | ✅ Verify |

**Success Criteria for Stage 0**:
- ✅ **100% pass required** - Any failure blocks further validation
- ✅ All data sources verified as real (no mock)
- ✅ All algorithms verified as complete (no simplified)
- ✅ All parameters traceable to sources
- ✅ Zero forbidden keywords found

**Output**: `results/validation/stage0_academic_compliance.json`

---

### Stage 1: Unit Validation

**Duration**: ~10 minutes
**Script**: `scripts/validation/stage1_unit_tests.py`

**Validation Items**:

#### 1.1 BaseAgent Protocol Compliance
```python
def test_baseagent_protocol():
    # DQNAgent implements all required methods
    agent = DQNAgent(obs_space, act_space, config)
    assert hasattr(agent, 'select_action')
    assert hasattr(agent, 'update')
    assert hasattr(agent, 'save')
    assert hasattr(agent, 'load')
    assert hasattr(agent, 'on_episode_start')
    assert hasattr(agent, 'on_episode_end')
```

#### 1.2 Strategy Protocol Compliance
```python
def test_strategy_protocol():
    # All strategies implement HandoverStrategy protocol
    from strategies import is_valid_strategy
    assert is_valid_strategy(StrongestRSRPStrategy())
    assert is_valid_strategy(A4BasedStrategy())
    assert is_valid_strategy(D2BasedStrategy())
    assert is_valid_strategy(DQNAgent(...))  # Duck typing
```

#### 1.3 ReplayBuffer Functionality
```python
def test_replay_buffer():
    buffer = ReplayBuffer(capacity=1000)

    # Store experience
    buffer.push(state, action, reward, next_state, done)
    assert len(buffer) == 1

    # Sample batch
    batch = buffer.sample(batch_size=32)
    assert len(batch) == 32

    # Capacity limit
    for i in range(1100):
        buffer.push(...)
    assert len(buffer) == 1000  # Max capacity
```

#### 1.4 Strategy Logic Correctness

**A4-based Strategy**:
```python
def test_a4_strategy_logic():
    strategy = A4BasedStrategy(threshold_dbm=-100.0, hysteresis_db=1.5)

    # Test case 1: No candidate exceeds threshold
    obs = create_observation(rsrp_values=[-110, -105, -108])  # All below -100
    action = strategy.select_action(obs, serving_satellite_idx=0)
    assert action == 0  # Stay

    # Test case 2: Candidate exceeds threshold and better than serving
    obs = create_observation(rsrp_values=[-105, -95, -98])  # sat 1,2 above -100
    action = strategy.select_action(obs, serving_satellite_idx=0)
    assert action == 2  # Handover to sat 1 (strongest among candidates)

    # Test case 3: Candidate exceeds threshold but worse than serving
    obs = create_observation(rsrp_values=[-90, -95, -98])  # Serving best
    action = strategy.select_action(obs, serving_satellite_idx=0)
    assert action == 0  # Stay
```

**D2-based Strategy**:
```python
def test_d2_strategy_logic():
    strategy = D2BasedStrategy(threshold1_km=1412.8, threshold2_km=1005.8)

    # Test case 1: Serving satellite not too far
    obs = create_observation(distances=[1000, 1200, 1100])  # Serving < threshold1
    action = strategy.select_action(obs, serving_satellite_idx=0)
    assert action == 0  # Stay (D2 condition 1 not met)

    # Test case 2: Serving too far, neighbor close enough
    obs = create_observation(distances=[1500, 900, 1100])  # Serving > 1412.8, sat 1 < 1005.8
    action = strategy.select_action(obs, serving_satellite_idx=0)
    assert action == 2  # Handover to sat 1 (closest among D2 candidates)

    # Test case 3: Serving too far, but no close neighbors
    obs = create_observation(distances=[1500, 1400, 1300])  # All above threshold2
    action = strategy.select_action(obs, serving_satellite_idx=0)
    assert action == 0  # Stay (D2 condition 2 not met)
```

#### 1.5 Multi-Level Config Loading
```python
def test_multi_level_configs():
    from configs import get_level_config, list_all_levels

    # All 6 levels loadable
    for level in range(6):
        config = get_level_config(level)
        assert 'num_satellites' in config
        assert 'num_episodes' in config
        assert 'estimated_time_hours' in config

    # Level 1 is recommended
    config = get_level_config(1)
    assert config['recommended'] == True
    assert config['num_satellites'] == 20
    assert config['num_episodes'] == 100

    # Invalid level raises error
    try:
        get_level_config(10)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
```

**Success Criteria**:
- ✅ All unit tests pass (100%)
- ✅ Protocol compliance verified
- ✅ Strategy logic matches standards
- ✅ Multi-level configs loadable

**Output**: `results/validation/stage1_unit_tests.json`

---

### Stage 2: Integration Validation

**Duration**: ~30 minutes
**Script**: `scripts/validation/stage2_integration_tests.py`

**Validation Items**:

#### 2.1 DQN + OffPolicyTrainer Integration
```python
def test_dqn_trainer_integration():
    env = create_test_env(num_satellites=10)
    agent = DQNAgent(env.observation_space, env.action_space, config)
    trainer = OffPolicyTrainer(env, agent, config)

    # Run 3 episodes
    for ep in range(3):
        metrics = trainer.train_episode(episode_idx=ep)
        assert 'episode_reward' in metrics
        assert 'loss' in metrics
        assert 'epsilon' in metrics

    # Epsilon should decay
    assert agent.epsilon < agent.epsilon_start
```

#### 2.2 Multi-Level Training Integration
```python
def test_multi_level_training():
    # Test Level 0 (smoke test)
    level_config = get_level_config(0)
    env = create_env(num_satellites=level_config['num_satellites'])
    agent = DQNAgent(...)
    trainer = OffPolicyTrainer(...)

    # Should complete without errors
    for ep in range(level_config['num_episodes']):
        metrics = trainer.train_episode(ep)

    # Verify metrics collected
    assert len(trainer.episode_rewards) == level_config['num_episodes']
```

#### 2.3 TensorBoard Logging
```python
def test_tensorboard_logging():
    import os
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir='results/validation/tensorboard_test')

    # Log training metrics
    writer.add_scalar('train/reward', 100.0, 0)
    writer.add_scalar('train/loss', 0.5, 0)
    writer.flush()

    # Verify log file created
    assert os.path.exists('results/validation/tensorboard_test')
```

#### 2.4 Checkpoint Save/Load
```python
def test_checkpoint_save_load():
    env = create_test_env(num_satellites=10)
    agent1 = DQNAgent(env.observation_space, env.action_space, config)

    # Train for a few steps
    for _ in range(100):
        state = env.reset()[0]
        action = agent1.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent1.update(state, action, reward, next_state, done)

    # Save checkpoint
    checkpoint_path = 'results/validation/test_checkpoint.pt'
    agent1.save(checkpoint_path)

    # Load into new agent
    agent2 = DQNAgent(env.observation_space, env.action_space, config)
    agent2.load(checkpoint_path)

    # Compare Q-network weights
    import torch
    for p1, p2 in zip(agent1.q_network.parameters(), agent2.q_network.parameters()):
        assert torch.allclose(p1, p2), "Weights should match after loading"
```

#### 2.5 Evaluation Framework Integration
```python
def test_evaluation_framework():
    from scripts.evaluate_strategies import evaluate_strategy, compare_strategies

    env = create_test_env(num_satellites=10)

    # Single strategy evaluation
    strategy = StrongestRSRPStrategy()
    metrics = evaluate_strategy(strategy, env, num_episodes=5, seed=42)
    assert 'avg_reward' in metrics
    assert 'handover_rate_pct' in metrics

    # Multi-strategy comparison
    strategies = {
        'Strongest RSRP': StrongestRSRPStrategy(),
        'A4-based': A4BasedStrategy(),
        'D2-based': D2BasedStrategy(),
    }
    df = compare_strategies(strategies, env, num_episodes=5, seed=42)
    assert len(df) == 3
    assert 'strategy' in df.columns
    assert 'avg_reward' in df.columns
```

**Success Criteria**:
- ✅ DQN training loop works end-to-end
- ✅ Multi-level configs work with training
- ✅ TensorBoard logging functional
- ✅ Checkpoints save/load correctly
- ✅ Evaluation framework works

**Output**: `results/validation/stage2_integration_tests.json`

---

### Stage 3: E2E Baseline Comparison (Level 1)

**Duration**: ~2 hours (100 episodes × 3 strategies)
**Script**: `scripts/validation/stage3_e2e_baseline_comparison.py`

**Validation Goal**: Verify all rule-based baselines work on Level 1 and produce expected performance patterns

**Setup**:
- Level: 1 (Quick Validation)
- Satellites: 20
- Episodes: 100
- Seed: 42 (reproducibility)

**Validation Items**:

#### 3.1 All Strategies Complete Successfully
```python
strategies = {
    'Strongest RSRP': StrongestRSRPStrategy(),
    'A4-based': A4BasedStrategy(threshold_dbm=-100.0, hysteresis_db=1.5),
    'D2-based': D2BasedStrategy(threshold1_km=1412.8, threshold2_km=1005.8),
}

# Run comparison
df = compare_strategies(strategies, env, num_episodes=100, seed=42)

# All strategies should complete
assert len(df) == 3
assert all(df['num_episodes'] == 100)
```

#### 3.2 Performance Pattern Validation
Expected pattern (from `results/EXPECTED_RESULTS.md`):
```
D2-based > A4-based > Strongest RSRP (in terms of avg_reward)
```

**Validation**:
```python
# Check ranking order
best = df.iloc[0]
worst = df.iloc[-1]

# Expected best: D2-based or A4-based (geometry vs RSRP tradeoff)
assert best['strategy'] in ['D2-based', 'A4-based']

# Expected worst: Strongest RSRP (no hysteresis)
assert worst['strategy'] == 'Strongest RSRP'

# Handover rate should be reasonable (4-10%)
for _, row in df.iterrows():
    assert 0 <= row['handover_rate_pct'] <= 20, f"Unreasonable HO rate: {row['handover_rate_pct']}"

# Ping-pong rate pattern (D2 should have lowest)
d2_row = df[df['strategy'] == 'D2-based'].iloc[0]
strongest_row = df[df['strategy'] == 'Strongest RSRP'].iloc[0]
assert d2_row['ping_pong_rate_pct'] <= strongest_row['ping_pong_rate_pct'], \
    "D2 should have lower ping-pong than Strongest RSRP"
```

#### 3.3 Metrics Completeness
```python
required_metrics = [
    'strategy',
    'avg_reward',
    'std_reward',
    'avg_handovers',
    'handover_rate_pct',
    'avg_ping_pongs',
    'ping_pong_rate_pct',
    'avg_rsrp_dbm',
]

for metric in required_metrics:
    assert metric in df.columns, f"Missing metric: {metric}"
```

#### 3.4 Reproducibility Check
```python
# Run twice with same seed
df1 = compare_strategies(strategies, env, num_episodes=10, seed=42)
df2 = compare_strategies(strategies, env, num_episodes=10, seed=42)

# Results should be identical
for col in ['avg_reward', 'handover_rate_pct']:
    assert np.allclose(df1[col], df2[col]), f"Non-reproducible results for {col}"
```

**Success Criteria**:
- ✅ All 3 strategies complete 100 episodes without errors
- ✅ Performance pattern matches expectations (D2/A4 > Strongest RSRP)
- ✅ Handover rates within reasonable range (4-10%)
- ✅ All metrics collected correctly
- ✅ Results reproducible with same seed

**Output**:
- `results/validation/stage3_baseline_comparison.json`
- `results/validation/stage3_baseline_comparison.csv`

---

### Stage 4: E2E DQN Training (Level 1)

**Duration**: ~2 hours (100 episodes)
**Script**: `scripts/validation/stage4_e2e_dqn_training.py`

**Validation Goal**: Verify DQN training works end-to-end on Level 1 and produces valid learning behavior

**Setup**:
- Level: 1 (Quick Validation)
- Satellites: 20
- Episodes: 100
- Seed: 42

**Validation Items**:

#### 4.1 Training Completion
```python
# Run training
python train.py --algorithm dqn --level 1 --episodes 100 --seed 42 \
    --output-dir results/validation/stage4_dqn

# Should complete without errors
assert training completed successfully
assert 100 episodes logged
```

#### 4.2 Learning Curve Validation
```python
# Load training metrics
import pandas as pd
metrics = pd.read_csv('results/validation/stage4_dqn/metrics.csv')

# Check for learning progress
early_reward = metrics['episode_reward'][:20].mean()
late_reward = metrics['episode_reward'][-20:].mean()

# Expect improvement (or at least stability)
# Note: 100 episodes may not show strong improvement, but should not degrade
assert late_reward >= early_reward - 50, \
    f"Significant degradation: early={early_reward}, late={late_reward}"
```

#### 4.3 Epsilon Decay Validation
```python
# Epsilon should decay over time
early_epsilon = metrics['epsilon'][:10].mean()
late_epsilon = metrics['epsilon'][-10:].mean()

assert late_epsilon < early_epsilon, "Epsilon should decay"
assert late_epsilon >= config['epsilon_end'], \
    f"Epsilon below minimum: {late_epsilon} < {config['epsilon_end']}"
```

#### 4.4 Loss Stability
```python
# Loss should not explode
assert metrics['loss'].max() < 1000, "Loss exploding"
assert not metrics['loss'].isna().any(), "Loss contains NaN"

# Loss should generally stabilize (or trend downward)
early_loss = metrics['loss'][20:40].mean()  # Skip first 20 (unstable)
late_loss = metrics['loss'][-20:].mean()
# Allow some variance, but should not increase drastically
assert late_loss < early_loss * 2, f"Loss increasing: {early_loss} -> {late_loss}"
```

#### 4.5 Checkpoint Validation
```python
# Verify checkpoints saved
import os
checkpoint_dir = 'results/validation/stage4_dqn/checkpoints'
assert os.path.exists(checkpoint_dir)

# Load final checkpoint
final_checkpoint = os.path.join(checkpoint_dir, 'final_model.pt')
assert os.path.exists(final_checkpoint)

# Test loaded agent
agent = DQNAgent(...)
agent.load(final_checkpoint)

# Agent should be able to select actions
env = create_env(...)
obs, _ = env.reset()
action = agent.select_action(obs, deterministic=True)
assert 0 <= action < env.action_space.n
```

#### 4.6 TensorBoard Validation
```python
# Verify TensorBoard logs exist
log_dir = 'results/validation/stage4_dqn/tensorboard'
assert os.path.exists(log_dir)

# Check key metrics logged
from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

assert 'train/reward' in ea.Tags()['scalars']
assert 'train/loss' in ea.Tags()['scalars']
assert 'train/epsilon' in ea.Tags()['scalars']
```

#### 4.7 Comparison with Rule-based Baselines
```python
# Evaluate trained DQN on same setup
metrics = evaluate_strategy(agent, env, num_episodes=100, seed=42)

# DQN may or may not outperform baselines after only 100 episodes,
# but should be in reasonable range
assert -500 <= metrics['avg_reward'] <= 500, \
    f"DQN reward out of reasonable range: {metrics['avg_reward']}"
assert 0 <= metrics['handover_rate_pct'] <= 20, \
    f"DQN handover rate unreasonable: {metrics['handover_rate_pct']}"
```

**Success Criteria**:
- ✅ Training completes 100 episodes without crashes
- ✅ Learning curve shows improvement or stability (no severe degradation)
- ✅ Epsilon decays properly from start to end
- ✅ Loss remains stable (no explosion, no NaN)
- ✅ Checkpoints save and load correctly
- ✅ TensorBoard logs all key metrics
- ✅ Trained agent produces reasonable performance

**Output**:
- `results/validation/stage4_dqn_training.json`
- `results/validation/stage4_dqn/metrics.csv`
- `results/validation/stage4_dqn/checkpoints/final_model.pt`
- `results/validation/stage4_dqn/tensorboard/` logs

---

## Validation Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│  FULL VALIDATION PIPELINE (~4-5 hours)                     │
└─────────────────────────────────────────────────────────────┘

Stage 0: Academic Compliance (P0 - CRITICAL)     [~5 min]
  ├─ Data source verification
  ├─ Algorithm completeness check
  ├─ Parameter traceability validation
  ├─ Forbidden content search
  └─ Standards compliance matrix
       ↓
   [GATE: 100% pass required or ABORT]
       ↓
Stage 1: Unit Validation                         [~10 min]
  ├─ Protocol compliance tests
  ├─ Strategy logic tests
  ├─ ReplayBuffer tests
  └─ Multi-level config tests
       ↓
   [GATE: All tests pass]
       ↓
Stage 2: Integration Validation                  [~30 min]
  ├─ DQN + Trainer integration
  ├─ Multi-level training integration
  ├─ TensorBoard logging test
  ├─ Checkpoint save/load test
  └─ Evaluation framework test
       ↓
   [GATE: All integrations work]
       ↓
Stage 3: E2E Baseline Comparison (Level 1)       [~2 hours]
  ├─ Run 3 strategies × 100 episodes
  ├─ Verify performance patterns
  ├─ Check metrics completeness
  └─ Validate reproducibility
       ↓
   [GATE: Baselines work correctly]
       ↓
Stage 4: E2E DQN Training (Level 1)             [~2 hours]
  ├─ Train DQN for 100 episodes
  ├─ Verify learning curve
  ├─ Validate epsilon decay
  ├─ Check loss stability
  ├─ Test checkpoint save/load
  └─ Compare with baselines
       ↓
   [GATE: DQN training functional]
       ↓
┌─────────────────────────────────────────────────────────────┐
│  VALIDATION COMPLETE - Generate Report                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Success Criteria Summary

### Stage 0 (Academic Compliance) - MANDATORY 100% PASS
| Check | Requirement | Status |
|-------|-------------|--------|
| No hardcoded data | All params from configs | ⏳ Validate |
| No mock data | All from orbit-engine/TLE | ⏳ Validate |
| No simplified algos | Full 3GPP/ITU-R/DQN impl | ⏳ Validate |
| Parameter traceability | All params have SOURCE | ⏳ Validate |
| Standards compliance | A4/D2/RSRP match specs | ⏳ Validate |
| Forbidden keywords | Zero "simplified/mock/fake" | ⏳ Validate |

**If ANY Stage 0 check fails → ABORT and fix immediately**

### Stage 1-4 (Functional) - Expected Pass Rates
- Stage 1 (Unit): 100% tests pass
- Stage 2 (Integration): 100% tests pass
- Stage 3 (Baseline E2E): All strategies complete, expected patterns
- Stage 4 (DQN E2E): Training completes, learning curve reasonable

---

## Output Artifacts

### Validation Results
```
results/validation/
├── stage0_academic_compliance.json      # P0 critical results
├── stage1_unit_tests.json
├── stage2_integration_tests.json
├── stage3_baseline_comparison.json
├── stage3_baseline_comparison.csv       # Raw baseline data
├── stage4_dqn_training.json
├── stage4_dqn/
│   ├── metrics.csv                      # DQN training metrics
│   ├── checkpoints/final_model.pt       # Trained model
│   └── tensorboard/                     # TensorBoard logs
└── validation_summary.json              # Overall summary
```

### Validation Report
```
docs/validation/
└── VALIDATION_REPORT.md                 # Human-readable report
```

---

## Execution Instructions

### Quick Validation (Stage 0 only - 5 min)
```bash
# P0 academic compliance check
python scripts/validation/stage0_academic_compliance.py
```

**Use case**: Quick check before starting any experiments

### Partial Validation (Stage 0-2 - 45 min)
```bash
# Run unit + integration tests
./scripts/validation/run_partial_validation.sh
```

**Use case**: After code changes, before long experiments

### Full Validation (All stages - 4-5 hours)
```bash
# Complete validation pipeline
./scripts/validation/run_full_validation.sh

# Output:
# - All stage results in results/validation/
# - Final report in docs/validation/VALIDATION_REPORT.md
```

**Use case**:
- Before submitting paper
- Before baseline experiments
- After major refactoring

---

## Validation Report Template

The validation report (`VALIDATION_REPORT.md`) will include:

1. **Executive Summary**
   - Overall pass/fail status
   - Critical findings
   - Recommendations

2. **Stage-by-Stage Results**
   - Stage 0: Academic compliance (PASS/FAIL with details)
   - Stage 1: Unit tests (pass rate, failed tests)
   - Stage 2: Integration tests (pass rate, issues)
   - Stage 3: Baseline comparison (results, patterns)
   - Stage 4: DQN training (learning curves, metrics)

3. **Academic Compliance Certificate**
   - Data sources verified
   - Algorithm standards verified
   - Parameter traceability verified
   - Ready for publication: YES/NO

4. **Performance Baselines**
   - Rule-based baseline results (Level 1)
   - DQN baseline results (Level 1)
   - Comparison table

5. **Known Issues & Limitations**
   - Any test failures
   - Any deviations from expected patterns
   - Recommendations for fixes

6. **Sign-off**
   - Validation date
   - Framework version
   - Ready for research: YES/NO

---

## Maintenance & Updates

### When to Re-validate

**Full validation required**:
- After any algorithm changes (DQN, A4, D2)
- After environment changes
- After orbit-engine data updates
- Before paper submission
- Before baseline experiments

**Partial validation (Stage 0-2) sufficient**:
- After config changes
- After minor bug fixes
- After documentation updates

**No validation needed**:
- Documentation-only changes
- Comment updates
- README changes

### Continuous Integration (Future)

Recommended CI/CD setup:
```yaml
# .github/workflows/validation.yml
on: [push, pull_request]

jobs:
  quick-validation:
    runs-on: ubuntu-latest
    steps:
      - Stage 0: Academic compliance (~5 min)
      - Stage 1: Unit tests (~10 min)
      - Stage 2: Integration tests (~30 min)

  nightly-full-validation:
    runs-on: ubuntu-latest
    schedule:
      - cron: '0 0 * * *'  # Daily at midnight
    steps:
      - Stage 0-4: Full validation (~4-5 hours)
```

---

## References

### Standards Documents
- 3GPP TS 38.331 v18.5.1 (RRC Protocol, A4/D2 Events)
- 3GPP TS 38.214 (Physical Layer Procedures, RSRP)
- ITU-R P.676 (Atmospheric Attenuation)
- SGP4 Orbital Mechanics (NORAD)

### Data Sources
- Space-Track.org (TLE data)
- orbit-engine Stage 4 (71-day analysis)
- Yu et al. 2022 ("Handover Performance Evaluation using A4 Event")

### Framework Documents
- `README.md` - Project overview
- `IMPLEMENTATION_PLAN.md` - Refactoring plan
- `PHASE1_COMPLETE.md` - Phase 1 completion report
- `PHASE2_COMPLETE.md` - Phase 2 completion report
- `results/EXPECTED_RESULTS.md` - Expected baseline performance

---

**Document Version**: 1.0
**Last Updated**: 2025-10-20
**Status**: Ready for implementation
**Estimated Total Time**: 4-5 hours (full validation)
