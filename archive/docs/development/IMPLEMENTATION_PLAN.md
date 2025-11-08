# Modular RL Framework Implementation Plan

**目标**: 建立包含 DQN 和 Rule-based baselines 的完整框架，作為未來算法對比的基礎

**总时长**: 2-3周
**优先级**: P0（核心功能）

---

## 📋 Implementation Roadmap

```
Week 1-2: Phase 1 - DQN Refactoring (Foundation)
    ↓
Week 3: Phase 2 - Rule-based Baselines ⭐
    ↓
Framework Complete! Ready for algorithm comparison
```

**目標**: 建立通用框架 + 完整的 baseline 集合（1 個 RL baseline + 3 個 rule-based baselines），作為未來新算法對比的基礎
**详见**: [BASELINE_ALGORITHMS.md](../algorithms/BASELINE_ALGORITHMS.md)

---

## 📅 Phase 1: DQN Refactoring (Week 1-2)

**目标**: 重构现有DQN到新架构，保持功能不变

### ⚠️ Critical Requirement: 保留 Multi-Level Training

**Multi-Level Training** 是系统的 **Novel Aspect #1**（见 README.md），必须完整保留。

**要求**:
- ✅ 所有 6 个训练级别（Level 0-5）必须在重构后可用
- ✅ train.py 必须支持 `--level {0,1,2,3,4,5}` 参数
- ✅ 时间估算必须准确（Level 1 ≈ 2h, Level 3 ≈ 10h, Level 5 ≈ 35h）

### 🎯 Success Criteria
- [ ] BaseAgent接口定义完成
- [ ] OffPolicyTrainer实现完成
- [ ] DQNAgent继承BaseAgent并通过测试
- [ ] **Multi-Level Training 完整保留（P0）** ⭐
- [ ] 训练结果与 `train_online_rl.py` 一致（相同seed）
- [ ] `train.py --algorithm dqn --level {0-5}` 全部可运行 ⭐

### 📝 Detailed Tasks

#### Task 1.1: Create BaseAgent Interface (2-3 hours)

**File**: `src/agents/base_agent.py`

```python
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state, deterministic=False):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
```

**Validation**:
```bash
python -c "from src.agents.base_agent import BaseAgent; print('✅ Import OK')"
```

---

#### Task 1.2: Create OffPolicyTrainer (4-6 hours)

**File**: `src/trainers/off_policy_trainer.py`

**Implementation**:
- Replay buffer management
- Per-step update logic
- Episode metrics collection

**Test**:
```python
# Unit test
def test_off_policy_trainer():
    env = SatelliteHandoverEnv(...)
    agent = DQNAgent(...)
    trainer = OffPolicyTrainer(env, agent, config)
    metrics = trainer.train_episode(0)
    assert 'reward' in metrics
    assert 'loss' in metrics
```

---

#### Task 1.3: Refactor DQNAgent (6-8 hours)

**Changes**:
```python
# Before (train_online_rl.py)
class DQNAgentV2:
    def __init__(self, state_dim, action_dim, config):
        ...

# After (src/agents/dqn/dqn_agent.py)
from src.agents.base_agent import BaseAgent

class DQNAgent(BaseAgent):
    def __init__(self, obs_space, action_space, config):
        super().__init__()
        self.state_dim = obs_space.shape[0]
        self.action_dim = action_space.n
        ...
```

**Migration Checklist**:
- [ ] 继承 BaseAgent
- [ ] 实现 select_action()
- [ ] 实现 update()
- [ ] 实现 save/load()
- [ ] 移动到 `src/agents/dqn/`

---

#### Task 1.4: Implement Multi-Level Training (3-4 hours) ⭐ P0

**Priority**: P0 - CRITICAL (Novel Aspect #1)

**Goal**: 保留 6 个训练级别，确保重构后完整可用

**File 1**: `src/configs/training_levels.py`

```python
"""
Multi-Level Training Strategy Configuration

Novel Aspect #1 from README.md - must be preserved during refactoring.
"""

TRAINING_LEVELS = {
    0: {
        'name': 'Smoke Test',
        'num_satellites': 10,
        'num_episodes': 10,
        'estimated_time': '10 minutes',
        'description': 'System verification - code runs without errors',
        'use_case': 'Debug, deployment verification',
    },
    1: {
        'name': 'Quick Validation',
        'num_satellites': 20,
        'num_episodes': 100,
        'estimated_time': '2 hours',
        'description': 'Verify training logic, observe learning curve',
        'use_case': 'Fast idea validation, hyperparameter testing',
    },
    2: {
        'name': 'Development',
        'num_satellites': 50,
        'num_episodes': 300,
        'estimated_time': '6 hours',
        'description': 'Debug hyperparameters and reward functions',
        'use_case': 'Development iteration',
    },
    3: {
        'name': 'Validation',
        'num_satellites': 101,
        'num_episodes': 500,
        'estimated_time': '10 hours',
        'description': 'Validate effectiveness - recommended starting point',
        'use_case': 'Paper draft experiments',
        'recommended': True,
    },
    4: {
        'name': 'Baseline',
        'num_satellites': 101,
        'num_episodes': 1000,
        'estimated_time': '21 hours',
        'description': 'Establish stable baseline',
        'use_case': 'Paper experiments',
    },
    5: {
        'name': 'Full Training',
        'num_satellites': 101,
        'num_episodes': 1700,
        'estimated_time': '35 hours',
        'description': 'Complete training',
        'use_case': 'Final paper experiments',
    },
}

def get_level_config(level: int) -> dict:
    """Get configuration for specific training level"""
    if level not in TRAINING_LEVELS:
        raise ValueError(f"Invalid level {level}. Must be 0-5.")
    return TRAINING_LEVELS[level]
```

**File 2**: Update `train.py` to support `--level`

```python
# In train.py
from src.configs.training_levels import get_level_config

parser.add_argument('--level', type=int, choices=[0,1,2,3,4,5],
                   required=True,
                   help='Training level (0=smoke test, 5=full training)')

# Apply level config
level_config = get_level_config(args.level)
print(f"Training Level {args.level}: {level_config['name']}")
print(f"  Estimated time: {level_config['estimated_time']}")
print(f"  Satellites: {level_config['num_satellites']}")
print(f"  Episodes: {level_config['num_episodes']}")

config['training'].update({
    'num_satellites': level_config['num_satellites'],
    'num_episodes': level_config['num_episodes'],
})
```

**Validation**:
```bash
# Test all 6 levels
for level in 0 1 2 3 4 5; do
    echo "Testing Level $level..."
    python train.py --algorithm dqn --level $level --output-dir output/test_level_$level
done
```

**Success Criteria**:
- [ ] `src/configs/training_levels.py` created with all 6 levels
- [ ] `train.py --level {0-5}` all executable
- [ ] Time estimates accurate (validate Level 1 ≈ 2h)
- [ ] Config correctly applied (check num_satellites, num_episodes)

---

#### Task 1.5: Create Unified train.py (4-6 hours)

**File**: `train.py`

**Features**:
- Algorithm registry (ALGORITHM_REGISTRY)
- Config loading (algorithm + training_level)
- Unified training loop
- Logging and checkpointing

**Test**:
```bash
# Test with DQN
python train.py --algorithm dqn --level 0 --output-dir output/test_dqn
```

---

#### Task 1.6: Validation & Comparison (4-6 hours)

**Goal**: 验证重构后功能完整，包括算法正确性和 multi-level 功能

**Validation Script**: `scripts/validate_dqn_refactor.py`

```python
import torch
import numpy as np

def compare_training_results():
    """Compare old vs new DQN training"""
    # Train with old system
    old_rewards = train_old_dqn(seed=42, episodes=100)

    # Train with new system
    new_rewards = train_new_dqn(seed=42, episodes=100)

    # Statistical comparison
    assert np.abs(np.mean(old_rewards - new_rewards)) < 1.0
    print("✅ Refactoring validation passed")

def validate_multi_level():
    """Validate all 6 training levels"""
    import time

    for level in [0, 1, 3]:  # Test critical levels
        print(f"Testing Level {level}...")
        start_time = time.time()

        # Run training
        os.system(f"python train.py --algorithm dqn --level {level} "
                  f"--output-dir output/validate_level_{level}")

        elapsed = (time.time() - start_time) / 3600  # hours

        # Check time estimates
        expected_times = {0: 0.17, 1: 2.0, 3: 10.0}  # hours
        tolerance = 0.2  # 20%

        assert abs(elapsed - expected_times[level]) / expected_times[level] < tolerance, \
            f"Level {level} time mismatch: {elapsed:.1f}h vs {expected_times[level]}h"

        print(f"✅ Level {level} validated: {elapsed:.1f}h")
```

**Acceptance Test**:
- [ ] Algorithm: Same seed → same reward curve (±5% tolerance)
- [ ] Algorithm: Same loss curve trend
- [ ] Algorithm: Checkpoint compatibility (can load old DQN models)
- [ ] **Multi-Level: All 6 levels executable** ⭐
- [ ] **Multi-Level: Time estimates accurate (Level 1 ≈ 2h)** ⭐
- [ ] **Multi-Level: Config correctly applied** ⭐

---

### ⏰ Phase 1 Timeline

| Task | Duration | Dependencies | Deliverable |
|------|----------|--------------|-------------|
| 1.1 BaseAgent | 2-3h | None | base_agent.py |
| 1.2 OffPolicyTrainer | 4-6h | Task 1.1 | off_policy_trainer.py |
| 1.3 Refactor DQNAgent | 6-8h | Task 1.1, 1.2 | dqn/dqn_agent.py |
| **1.4 Multi-Level (P0)** ⭐ | **3-4h** | **None** | **training_levels.py** |
| 1.5 Unified train.py | 4-6h | Task 1.3, 1.4 | train.py |
| 1.6 Validation | 4-6h | All above | Validated refactor |
| 1.4 train.py | 4-6h | Task 1.3 | train.py |
| 1.5 Validation | 2-4h | Task 1.4 | validation_report.md |
| **Total** | **18-27h** | | **~3-4 days** |

---

## 📅 Phase 2: Rule-based Baselines (Week 3)

**目标**: 實現 rule-based handover baselines，建立完整的對比框架

**Priority**: P0 (框架必需)

**時間**: 1.5 days (7.5-9.5 hours)

**📖 詳細文檔**: [PHASE2_RULE_BASED_METHODS.md](PHASE2_RULE_BASED_METHODS.md)

---

### 📝 Tasks Summary

本 Phase 詳細內容請參考獨立文檔 [PHASE2_RULE_BASED_METHODS.md](PHASE2_RULE_BASED_METHODS.md)。

**Task 2.1**: Base Strategy Protocol (1h)
- 定義 `HandoverStrategy` Protocol
- Duck typing 設計（不繼承 BaseAgent）
- 與 RL framework zero coupling

**Task 2.2**: Implement 3 Strategies (3-4.5h)
1. **Strongest RSRP** (15 min) - 簡單啟發式策略
2. **A4-based Strategy** (30 min) - 基於 3GPP A4 Event 的完整策略，Yu 2022 驗證
3. **D2-based Strategy** (45 min) - 基於 3GPP D2 Event 的 NTN 專用策略 ⭐

**Note**: A4/D2 Event 是 3GPP 測量報告觸發條件，我們補充選擇邏輯和切換決策

**Task 2.3**: Unified Evaluation Framework (2h)
- 統一接口 `evaluate_strategy()`
- 支持 RL 和 rule-based 方法
- Level 1 快速測試

**Task 2.4**: Level 1 Comparison (2-3h)
- 對比 4 個方法：
  - **RL**: DQN
  - **Rule-based**: Strongest RSRP, A4-based Strategy, D2-based Strategy
- 生成對比報告，建立 baseline 框架

---

### 🎯 Success Criteria
- [ ] 3 個 rule-based baselines 實現完成
- [ ] Duck typing 設計（不依賴 BaseAgent）
- [ ] Level 1 測試通過（所有 4 個方法可運行）
- [ ] 統一評估框架完成（支持 RL + rule-based）
- [ ] **Baseline 框架完整，可用於未來算法對比** ⭐

---

### ⏰ Phase 2 Timeline

| Task | Duration | Dependencies | Deliverable |
|------|----------|--------------|-------------|
| 2.1 Base Protocol | 1h | Phase 1 | base_strategy.py |
| 2.2 3 Strategies | 3-4.5h | Task 2.1 | 3 strategy implementations |
| 2.3 Evaluation Framework | 2h | Task 2.2 | evaluate.py |
| 2.4 Level 1 Comparison | 2-3h | Task 2.3 | comparison_report.md |
| **Total** | **7.5-9.5h** | | **~1.5 days**

---

## 📊 Milestone Checklist

### After Phase 1
- [ ] BaseAgent interface finalized
- [ ] OffPolicyTrainer tested
- [ ] DQNAgent refactored and validated
- [ ] **Multi-Level Training (Level 0-5) working** ⭐
- [ ] train.py working with `--algorithm dqn --level {0-5}`
- [ ] Documentation updated

### After Phase 2 ⭐ COMPLETE
- [ ] 3 rule-based baselines implemented
- [ ] Unified evaluation framework working
- [ ] Level 1 baseline evaluation (1 RL + 3 rule-based) completed
- [ ] **Baseline framework complete and ready for algorithm comparison** ⭐

### Final Deliverables
- [ ] Unified train.py supporting DQN
- [ ] 3 rule-based baselines (Strongest RSRP, A4-based, D2-based)
- [ ] Unified evaluation framework (RL + rule-based)
- [ ] Complete documentation
- [ ] Unit tests for all components
- [ ] **Framework ready for future algorithm comparison** ⭐

---

## 🔧 Development Guidelines

### Code Quality Standards
- **Type hints**: All functions
- **Docstrings**: Google style
- **Tests**: Pytest for all agents
- **Linting**: Black + flake8

### Git Workflow
```bash
# Feature branch for each phase
git checkout -b phase1/dqn-refactor
git checkout -b phase2/ppo-implementation
git checkout -b phase3/dqn-variants
```

### Testing Strategy
- **Unit tests**: Each agent class
- **Integration tests**: Full training loop
- **Regression tests**: Compare with old DQN
- **Level 0 smoke test**: All new algorithms

---

## 🚨 Risk Mitigation

### Risk 1: Refactored DQN doesn't match old performance
**Mitigation**:
- Use exact same hyperparameters
- Compare with frozen random seeds
- Debug with small-scale tests first

### Risk 2: PPO doesn't converge
**Mitigation**:
- Start with proven hyperparameters from literature
- Test on simpler environments first (CartPole)
- Consult Stable-Baselines3 implementation

### Risk 3: Timeline延期
**Mitigation**:
- Phase 1-2 are complete project scope (no Phase 3-4)
- Minimum viable: DQN + 3 rule-based baselines
- Can reduce Level 1 experiment scope if time-constrained
- Track progress with regular checkpoints

---

## 📈 Success Metrics

### Quantitative
- [ ] All algorithms converge within expected episodes
- [ ] PPO shows lower variance than DQN (literature validation)
- [ ] Code coverage > 80%
- [ ] Training time per episode < 5s

### Qualitative
- [ ] Code is modular and extensible
- [ ] Documentation is clear
- [ ] Easy to add new algorithms (validated by Phase 2-3)
- [ ] Team consensus on architecture

---

## 📚 Reference Implementations

**Study these for inspiration** (don't copy directly):
1. **Stable-Baselines3**: https://github.com/DLR-RM/stable-baselines3
   - Best practices for RL implementations
2. **CleanRL**: https://github.com/vwxyzjn/cleanrl
   - Simple, educational implementations
3. **Tianshou**: https://github.com/thu-ml/tianshou
   - Modular RL framework design

---

**Created**: 2025-10-19
**Last Updated**: 2025-10-20 (建立 Baseline 框架)
**Status**: Planning Complete, Ready to Start Phase 1
**Estimated Total Time**:
- Phase 1 (DQN Refactoring): 2 weeks
- Phase 2 (Rule-based Baselines): 1.5 days
- **Total**: **2-3 weeks** ⭐

**Next Action**: Start Task 1.1 - Create BaseAgent Interface

**目標**: 建立包含 DQN (RL baseline) 和 3 個 rule-based baselines 的完整框架，作為未來算法對比的基礎。

**Note**: Phase 2 (Rule-based Baselines) 完全獨立於 RL 框架，實現快速（1.5 天）。框架完成後可輕鬆加入新算法進行對比。
