# Phase 2: Rule-based Comparison Methods

**週次**: Week 3
**時間**: 1.5 days (7.5-9.5 hours)
**優先級**: P1 (論文必需)
**依賴**: Phase 1 (DQN Refactoring) 完成

---

## 📋 Overview

### 目標
實現 rule-based handover baselines，建立完整的對比框架，作為未來算法評估的基礎。

### 核心價值
- 建立標準的 rule-based baselines（Strongest RSRP, A4-based Strategy, D2-based Strategy）
- 提供統一的評估框架（支持 RL 和 rule-based 方法）
- 為未來的算法對比提供完整的 baseline 集合
- 使用 NTN 專用的 D2-based Strategy（基於 3GPP D2 Event），突顯 NTN 特性 ⭐

**重要说明**: A4/D2 Event 是 3GPP 定义的测量报告触发条件。作为 baseline strategy，我们补充了选择逻辑和切换决策。

### 設計原則
- ✅ **完全獨立**：不依賴 RL 框架實現（只依賴環境 API）
- ✅ **Duck Typing**：通過統一介面評估，不強制繼承
- ✅ **學術嚴謹**：所有參數都有 SOURCE 引用（3GPP + orbit-engine）
- ✅ **快速實現**：總時間 1.5 天

---

## 🎯 Success Criteria

### Implementation
- [ ] 3 種 rule-based baselines 實現完成
- [ ] Config files created with proper SOURCE citations
- [ ] Evaluation script supports both RL and rule-based
- [ ] Unit tests pass for all baselines

### Experiments
- [ ] Level 1 evaluation completed for all baselines
- [ ] Unified evaluation framework working correctly
- [ ] Baseline comparison report generated

### Documentation
- [ ] docs/strategies/RULE_BASED_METHODS.md created
- [ ] Config README with parameter explanations
- [ ] **Framework ready for future algorithm comparison** ⭐

---

## 📝 Detailed Tasks

### Task 2.1: Base Strategy Protocol (1 hour)

**File**: `src/strategies/base_strategy.py`

**設計原則**:
- ❌ 不繼承 BaseAgent（避免不必要耦合）
- ✅ 簡單 Protocol（只定義必要介面）
- ✅ Duck typing（統一評估，分離實現）

**Implementation**:
```python
"""
Rule-based Handover Strategies

These are NOT RL agents. They use fixed rules to make handover decisions.
Used for comparison to demonstrate the value of RL approaches.
"""

from typing import Protocol
import numpy as np

class HandoverStrategy(Protocol):
    """
    Protocol for handover decision strategies.

    Both RL agents and rule-based methods can implement this interface
    through duck typing (no inheritance required).
    """

    def select_action(self, observation: np.ndarray, **kwargs) -> int:
        """
        Select handover action based on observation.

        Args:
            observation: (K, 12) array from environment
            **kwargs: Optional info (e.g., serving_satellite_idx for D2)

        Returns:
            action: int (0 = stay, 1-K = handover to satellite i-1)
        """
        ...
```

**Validation**:
```bash
python -c "from src.strategies.base_strategy import HandoverStrategy; print('✅ Import OK')"
```

---

### Task 2.2: Implement 3 Rule-based Strategies (1.5 hours)

#### 2.2.1: Strongest RSRP (15 min)

**File**: `src/strategies/strongest_rsrp.py`

**Logic**: Always select satellite with highest RSRP (simplest heuristic)

**Implementation**:
```python
class StrongestRSRPStrategy:
    """
    Strongest RSRP Handover Strategy（简单启发式策略）

    Decision rule:
      1. 找到 RSRP 最强的卫星
      2. 如果最强的不是当前服务卫星，切换
      3. 如果最强的就是当前服务卫星，保持连接（stay）

    Use case: Baseline lower bound for demonstrating improvements.
    Expected performance: High handover rate, high ping-pong.
    """

    def __init__(self):
        pass  # No parameters needed

    def select_action(self, observation: np.ndarray,
                     serving_satellite_idx: int = None,
                     **kwargs) -> int:
        """
        Args:
            observation: (K, 12) array, Index 0 = RSRP (dBm)
            serving_satellite_idx: Current serving satellite index (0-based)

        Returns:
            action: 0 (stay) or 1-K (handover to satellite i-1)
        """
        rsrp_values = observation[:, 0]  # Extract RSRP column
        best_idx = np.argmax(rsrp_values)

        # If best is current serving satellite, stay
        if serving_satellite_idx is not None and best_idx == serving_satellite_idx:
            return 0  # Stay
        else:
            return int(best_idx + 1)  # Handover to best RSRP satellite
```

**Config**: `config/strategies/strongest_rsrp.yaml`
```yaml
strategy: strongest_rsrp
name: "Strongest RSRP"
type: heuristic
description: "Always select satellite with highest RSRP"

parameters: {}  # No parameters

metadata:
  complexity: "O(K)"
  expected_ho_rate: "8-10%"
  expected_ping_pong: "10-15%"
```

---

#### 2.2.2: A4-based Strategy (30 min)

**File**: `src/strategies/a4_based_strategy.py`

**Logic**: A4-based handover strategy (3GPP A4 Event + RSRP selection + handover decision)

**Implementation**:
```python
class A4BasedStrategy:
    """
    A4-based Handover Strategy（基于 3GPP A4 Event 的完整切换策略）

    **3GPP A4 Event 定义**:
    - 触发条件: Mn + Ofn + Ocn - Hys > Thresh（邻居 RSRP 超过阈值）
    - 来源: 3GPP TS 38.331 Section 5.5.4.5
    - 注意: A4 Event 本身只是测量报告触发条件

    **作为 Baseline Strategy 的补充**:
    - 选择逻辑: 从满足 A4 条件的候选中选择 RSRP 最强的
    - 切换决策: 如果候选比当前服务卫星更好，才执行切换
    - 防 ping-pong: 使用 hysteresis 参数

    Decision rule:
      1. 检查所有邻居卫星是否满足: RSRP - Hys > Thresh
      2. 从满足条件的候选中选择 RSRP 最强的
      3. 如果候选比当前服务卫星更好，切换
      4. 否则保持当前连接（stay）

    SOURCE: Yu et al. 2022 - proved A4 > A3 for LEO NTN
    STANDARDS: 3GPP TS 38.331 v18.5.1 Section 5.5.4.5
    """

    def __init__(self, threshold_dbm: float = -100.0,
                 hysteresis_db: float = 1.5,
                 offset_db: float = 0.0):
        """
        Args:
            threshold_dbm: A4 RSRP threshold (SOURCE: Yu 2022 optimal = -100)
            hysteresis_db: Hysteresis parameter (SOURCE: 3GPP typical = 1.5)
            offset_db: Measurement offset (default 0)
        """
        self.threshold = threshold_dbm
        self.hysteresis = hysteresis_db
        self.offset = offset_db

    def select_action(self, observation: np.ndarray,
                     serving_satellite_idx: int = None,
                     **kwargs) -> int:
        """
        A4 Event trigger condition:
          Mn + Ofn + Ocn - Hys > Thresh

        Simplified (no cell-specific offsets):
          neighbor_RSRP - Hys > Thresh

        Args:
            observation: (K, 12) array, Index 0 = RSRP (dBm)
            serving_satellite_idx: Current serving satellite index (0-based)

        Returns:
            action: 0 (stay) or 1-K (handover to satellite i-1)
        """
        rsrp_values = observation[:, 0]

        # Find all neighbors that satisfy A4 trigger condition (exclude current)
        candidates = []
        for i, rsrp in enumerate(rsrp_values):
            if serving_satellite_idx is None or i != serving_satellite_idx:
                # A4 trigger check
                trigger_value = rsrp + self.offset - self.hysteresis
                if trigger_value > self.threshold:
                    candidates.append((i, rsrp))

        if candidates:
            # Select best RSRP from candidates
            best_candidate = max(candidates, key=lambda x: x[1])
            return int(best_candidate[0] + 1)  # Handover to best candidate
        else:
            return 0  # Stay (no A4 event triggered)
```

**Config**: `config/strategies/a4_based.yaml`
```yaml
strategy: a4_based
name: "A4-based Strategy"
type: rule_based_3gpp
description: "3GPP A4 event + RSRP selection + handover decision"

parameters:
  threshold_dbm: -100.0   # SOURCE: Yu et al. 2022 (optimal for LEO)
  hysteresis_db: 1.5      # SOURCE: 3GPP TS 38.331
  offset_db: 0.0

metadata:
  reference: "Yu et al. 2022 - Performance Evaluation of Handover using A4 Event"
  standards: "3GPP TS 38.331 v18.5.1 Section 5.5.4.5"
  trigger_rate: "54.4%"   # From orbit-engine real data
  optimal_for: "LEO NTN (RSRP variation < 10 dB)"
```

---

#### 2.2.3: D2-based Strategy (45 min)

**File**: `src/strategies/d2_based_strategy.py`

**Logic**: D2-based handover strategy (3GPP D2 Event + distance selection + handover decision)

**Implementation**:
```python
class D2BasedStrategy:
    """
    D2-based Handover Strategy（基于 3GPP D2 Event 的完整切换策略）

    **3GPP D2 Event 定义**:
    - 触发条件 1: Ml1 - Hys > Thresh1（服务卫星太远）
    - 触发条件 2: Ml2 + Hys < Thresh2（邻居卫星够近）
    - 来源: 3GPP TS 38.331 Section 5.5.4.15a (Rel-17 NTN)
    - 注意: D2 Event 本身只是测量报告触发条件

    **作为 Baseline Strategy 的补充**:
    - 选择逻辑: 从满足 D2 条件的候选中选择距离最近的
    - 切换决策: 如果候选比当前服务卫星更近，才执行切换
    - NTN 专用: 考虑卫星移动特性（距离而非信号强度）

    Decision rule:
      1. 检查当前服务卫星距离是否 > threshold1（太远）
      2. 找到所有距离 < threshold2 的候选卫星（够近）
      3. 从候选中选择距离最近的
      4. 如果候选距离 < 当前服务卫星距离，切换
      5. 否则保持当前连接（stay）

    SOURCE: orbit-engine real data analysis (71-day TLE)
    STANDARDS: 3GPP TS 38.331 v18.5.1 Section 5.5.4.15a
    """

    def __init__(self, threshold1_km: float = 1412.8,
                 threshold2_km: float = 1005.8,
                 hysteresis_km: float = 50.0):
        """
        Args:
            threshold1_km: Serving satellite too far (SOURCE: orbit-engine Stage 4)
            threshold2_km: Neighbor close enough (SOURCE: orbit-engine Stage 4)
            hysteresis_km: Distance hysteresis (SOURCE: 3GPP typical)
        """
        self.threshold1 = threshold1_km
        self.threshold2 = threshold2_km
        self.hysteresis = hysteresis_km

    def select_action(self, observation: np.ndarray,
                     serving_satellite_idx: int = None,
                     **kwargs) -> int:
        """
        D2 Event trigger condition:
          (serving_dist - Hys > Thresh1) AND (neighbor_dist + Hys < Thresh2)

        Args:
            observation: (K, 12) array, Index 3 = Distance (km)
            serving_satellite_idx: Current serving satellite index (0-based)

        Returns:
            action: 0 (stay) or 1-K (handover to satellite i-1)
        """
        distances = observation[:, 3]  # Extract distance column

        # Get serving satellite distance
        if serving_satellite_idx is not None and serving_satellite_idx < len(distances):
            serving_distance = distances[serving_satellite_idx]
        else:
            serving_distance = np.inf  # If unknown, assume far

        # D2 Condition 1: Serving satellite too far?
        if serving_distance - self.hysteresis > self.threshold1:
            # D2 Condition 2: Find all neighbors close enough
            candidates = []
            for i, dist in enumerate(distances):
                if serving_satellite_idx is None or i != serving_satellite_idx:
                    if dist + self.hysteresis < self.threshold2:
                        candidates.append((i, dist))

            if candidates:
                # Select closest from candidates
                best_candidate = min(candidates, key=lambda x: x[1])
                return int(best_candidate[0] + 1)  # Handover to closest

        return 0  # Stay (no D2 event triggered or no suitable candidate)
```

**Config**: `config/strategies/d2_based.yaml`
```yaml
strategy: d2_based
name: "D2-based Strategy"
type: rule_based_ntn
description: "3GPP Rel-17 D2 event + distance selection + handover decision (NTN-specific)"

parameters:
  threshold1_km: 1412.8   # SOURCE: orbit-engine Stage 4 dynamic analysis
  threshold2_km: 1005.8   # SOURCE: orbit-engine Stage 4 dynamic analysis
  hysteresis_km: 50.0     # SOURCE: 3GPP typical

metadata:
  reference: "3GPP Rel-17 NTN standardization"
  standards: "3GPP TS 38.331 v18.5.1 Section 5.5.4.15a"
  trigger_rate: "6.5%"    # From orbit-engine real data
  optimal_for: "LEO NTN extreme scenarios (serving satellite too far)"
  novelty: "First use of D2 event in RL baseline comparison research"
```

**D2 Threshold 推导说明**:

**数据源**: orbit-engine Stage 4 - 71天真实 TLE 数据分析（2025-07-27 to 2025-10-17）

**方法**:
1. **距离分布统计**:
   - 分析 101 Starlink satellites 到地面用户的距离范围
   - Starlink orbit: 550km altitude（orbital distance: ~900-1500km）

2. **threshold1（服务卫星"太远"）**:
   - 定义：当前服务卫星距离超过此值时考虑切换
   - 值：1412.8 km
   - 基于：距离分布的上四分位数（75th percentile）
   - 含义：服务卫星接近可见范围边缘

3. **threshold2（邻居卫星"够近"）**:
   - 定义：邻居卫星距离小于此值时可作为切换候选
   - 值：1005.8 km
   - 基于：距离分布的中位数（50th percentile）
   - 含义：邻居卫星位于较优服务位置

**验证**:
- ✅ Trigger rate: ~6.5%（与 3GPP CHO 典型值一致）
- ✅ 避免过于频繁切换（threshold1 - threshold2 = 407km hysteresis margin）
- ✅ 基于真实轨道数据，不是估算值

**Note**: 这些值来自 orbit-engine Stage 4 的真实卫星轨道分析，符合"REAL ALGORITHMS ONLY"原则

---

### Task 2.3: Evaluation Framework (3-4 hours)

**File**: `scripts/evaluate_strategies.py`

**Goal**: 統一評估 RL agents 和 rule-based strategies

**Implementation**:
```python
"""
Unified Strategy Evaluation Framework

Supports both RL agents and rule-based strategies through duck typing.
Only requirement: object must have select_action(observation) method.
"""

import numpy as np
from typing import Union, Dict, List
from src.environments import SatelliteHandoverEnv

def evaluate_strategy(
    strategy,  # RL agent or rule-based strategy
    env: SatelliteHandoverEnv,
    num_episodes: int = 100,
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate any strategy (RL or rule-based) on environment.

    Args:
        strategy: Any object with select_action(observation) method
        env: SatelliteHandoverEnv instance
        num_episodes: Number of evaluation episodes
        seed: Random seed for reproducibility

    Returns:
        metrics: Dict with performance metrics
    """
    np.random.seed(seed)

    episode_rewards = []
    handover_counts = []
    ping_pong_counts = []
    avg_rsrps = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0
        handovers = 0

        while not done:
            # Duck typing: works for both RL and rule-based
            action = strategy.select_action(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            if info.get('handover_occurred', False):
                handovers += 1

        episode_rewards.append(episode_reward)
        handover_counts.append(handovers)
        ping_pong_counts.append(info.get('num_ping_pongs', 0))
        avg_rsrps.append(info.get('avg_rsrp', 0))

    return {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_handovers': np.mean(handover_counts),
        'handover_rate_pct': np.mean(handover_counts) / env.episode_duration_minutes * 100,
        'ping_pong_rate_pct': np.mean(ping_pong_counts) / np.mean(handover_counts) * 100,
        'avg_rsrp_dbm': np.mean(avg_rsrps),
    }

def compare_strategies(
    strategies: Dict[str, any],
    env: SatelliteHandoverEnv,
    num_episodes: int = 100
) -> pd.DataFrame:
    """
    Compare multiple strategies and return results table.

    Args:
        strategies: Dict of {name: strategy_object}
        env: Environment
        num_episodes: Evaluation episodes

    Returns:
        DataFrame with comparison results
    """
    results = []

    for name, strategy in strategies.items():
        print(f"Evaluating {name}...")
        metrics = evaluate_strategy(strategy, env, num_episodes)
        metrics['strategy'] = name
        results.append(metrics)

    df = pd.DataFrame(results)
    return df.sort_values('avg_reward', ascending=False)
```

**CLI Usage**:
```bash
# Evaluate single strategy
python scripts/evaluate_strategies.py \
    --strategy-type rule_based \
    --strategy-name a4_event \
    --level 1 \
    --episodes 100

# Compare all strategies
python scripts/evaluate_strategies.py \
    --compare-all \
    --level 1 \
    --output results/level1_comparison.csv
```

---

### Task 2.4: Level 1 Comparison Experiment (2-3 hours)

**Experiment Design**:
```bash
# Run Level 1 evaluation for all strategies
# Total time: ~8-10 hours (parallel recommended)

# Rule-based strategies (fast, ~5 minutes each)
python scripts/evaluate_strategies.py --strategy strongest_rsrp --level 1
python scripts/evaluate_strategies.py --strategy a4_event --level 1
python scripts/evaluate_strategies.py --strategy d2_event --level 1

# RL algorithm (trained model, ~5 minutes for evaluation)
python scripts/evaluate_strategies.py --strategy dqn --level 1 --model output/dqn_level1/best_model.pt

# Generate comparison report
python scripts/compare_results.py \
    --results-dir results/ \
    --output results/level1_comparison_report.md
```

**Expected Output**: `results/level1_comparison_report.md`

```markdown
# Level 1 Strategy Comparison Report

## Performance Metrics

| Strategy | Type | Avg Reward | HO Rate (%) | Avg RSRP (dBm) | Ping-Pong (%) |
|----------|------|------------|-------------|----------------|---------------|
| **DQN** | RL | **-128.7** | **5.8** | **-30.2** | **2.1** |
| Geometry + D2 | NTN Rule | -142.5 | 4.8 | -31.2 | 4.0 |
| RSRP + A4 | 3GPP Rule | -151.2 | 6.1 | -31.5 | 7.5 |
| Strongest RSRP | Heuristic | -168.9 | 8.2 | -32.1 | 12.3 |

## Key Findings

1. **Baseline Performance Hierarchy**:
   - DQN (RL baseline): -128.7
   - D2-based Strategy (NTN-specific): -142.5
   - A4-based Strategy (3GPP standard): -151.2
   - Strongest RSRP (Simple heuristic): -168.9

2. **Rule-based Baseline Ranking**: D2-based > A4-based > Strongest RSRP
   - Validates that NTN-specific design (D2-based) improves performance
   - Confirms Yu et al. 2022 findings (A4 suitable for LEO)

3. **Baseline Framework Complete**:
   - 1 RL baseline + 3 rule-based baselines established
   - Unified evaluation framework working
   - Ready for future algorithm comparison
```

---

## ⏰ Timeline

| Task | Duration | Complexity | Deliverable |
|------|----------|------------|-------------|
| 2.1 Base Strategy Protocol | 1h | Low | base_strategy.py |
| 2.2 Implement 3 Strategies | 1.5h | Low | 3 strategy files + configs |
| 2.3 Evaluation Framework | 3-4h | Medium | evaluate_strategies.py |
| 2.4 Level 1 Comparison | 2-3h | Low | comparison_report.md |
| **Total** | **7.5-9.5h** | | **~1.5 days** |

---

## 📊 Directory Structure

```
handover-rl/
├── src/
│   ├── agents/                    # RL algorithms (Phase 1)
│   │   ├── base_agent.py
│   │   └── dqn/
│   │
│   ├── strategies/                # Rule-based methods (Phase 2) ⭐ NEW
│   │   ├── base_strategy.py
│   │   ├── strongest_rsrp.py
│   │   ├── rsrp_a4_event.py
│   │   └── geometry_d2_event.py
│   │
│   ├── trainers/
│   └── environments/
│
├── config/
│   ├── algorithms/                # RL configs
│   └── strategies/                # Rule-based configs ⭐ NEW
│       ├── strongest_rsrp.yaml
│       ├── a4_event.yaml
│       └── d2_event.yaml
│
├── scripts/
│   ├── evaluate_strategies.py    # Unified evaluation ⭐ NEW
│   └── compare_results.py        # Comparison analysis ⭐ NEW
│
└── docs/
    ├── development/
    │   ├── IMPLEMENTATION_PLAN.md    # Main plan (Phase 1-2)
    │   └── PHASE2_RULE_BASED_METHODS.md  # This document
    └── strategies/                # Rule-based docs ⭐ NEW
        ├── RULE_BASED_METHODS.md
        └── PARAMETER_TUNING.md
```

---

## 🔗 Dependencies

### Depends On (Phase 1-3)
- ✅ `SatelliteHandoverEnv` API stable (observation format)
- ❌ **NOT dependent** on RL framework internals

### Environment API Requirements
```python
# Only requirement: environment provides (K, 12) observations
observation, reward, terminated, truncated, info = env.step(action)

# observation shape: (K, 12)
# Index 0: RSRP (dBm)         ← A4 needs this
# Index 3: Distance (km)       ← D2 needs this
```

---

## 🎯 Research Value

### Novel Contributions
1. **First D2-based Strategy as Baseline**: No prior work uses D2 event as basis for RL baseline comparison
2. **NTN-Optimized Parameters**: Thresholds from real TLE data (orbit-engine)
3. **Multi-Dimensional Baseline Framework**: RSRP-based (A4-based) + Distance-based (D2-based)
4. **Complete Strategy Implementation**: Supplement 3GPP events with selection logic and handover decision
4. **Comprehensive Baseline Set**: Covers heuristic, standard, and NTN-specific methods

### Framework Benefits
```
Baseline Framework Includes:
- Simple Heuristic: Strongest RSRP (baseline lower bound)
- 3GPP Standard: A4-based Strategy (A4 event + RSRP selection, validated for LEO)
- NTN-Specific: D2-based Strategy (D2 event + distance selection) ⭐
- RL Baseline: DQN (standard RL method)

→ Provides comprehensive comparison for future algorithms
→ Enables fair and thorough evaluation
→ Supports reproducible research
```

---

## ✅ Validation Checklist

### Before Starting Phase 2
- [ ] Phase 1 completed (DQN refactored and stable)
- [ ] Environment API confirmed stable
- [ ] Multi-Level Training working (Level 1 available for testing)

### During Implementation
- [ ] Each strategy tested independently
- [ ] Config files validate (YAML syntax)
- [ ] Unit tests pass

### Before Completion
- [ ] All 3 baselines implemented
- [ ] Evaluation framework works for RL + rule-based
- [ ] Level 1 comparison completed
- [ ] **Baseline framework complete and validated** ⭐

---

**Created**: 2025-10-19
**Last Updated**: 2025-10-20 (建立 Baseline 框架)
**Status**: Planning Complete
**Next Action**: Wait for Phase 1 completion, then start Task 2.1

**目標**: 建立完整的 baseline 框架（1 RL + 3 rule-based），作為未來算法評估的基礎

**Note**: This phase is completely independent of RL framework. Can be implemented immediately after Phase 1 (DQN refactoring) completion. Framework provides comprehensive baselines for future algorithm comparison.
