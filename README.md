# Handover-RL: LEO Satellite Handover Optimization with RL

**Modular reinforcement learning framework for optimizing LEO satellite handover**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0+-green.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Current Status

- ✅ **BC Training Complete**: 88.81% accuracy (target: 85-95%) - See [BC V4 Report](docs/reports/TRAINING_REPORT_V4_FINAL.md)
- ✅ **Data Leakage Fixed**: Eliminated 100% accuracy problem - See [Diagnosis](docs/reports/DIAGNOSIS_100_ACCURACY.md)
- ✅ **Threshold Design**: Data-driven (-34.5 dBm) - See [Recommendations](docs/reports/FINAL_THRESHOLD_RECOMMENDATIONS.md)
- 📍 **Next**: DQN Training with BC warm-start - See [Project Status](docs/PROJECT_STATUS.md)
- ✅ **Gymnasium Environment**: Standards-compliant, algorithm-agnostic
- ✅ **Multi-Level Training**: 10 minutes → 35 hours progressive strategy

**Last Updated**: 2025-10-21

---

## 🚀 Quick Start

### Prerequisites

**Software**:
- Python 3.10+
- PyTorch 2.0+
- orbit-engine installed at `../orbit-engine`

**Hardware**:
- **RAM**: 8GB+ (16GB recommended for Level 3+)
- **CPU**: Multi-core processor (4+ cores recommended)
- **GPU**: Optional but recommended for faster training
  - CUDA-capable GPU with 4GB+ VRAM (e.g., NVIDIA GTX 1650 or better)
  - CPU-only training supported but slower
- **Storage**: 2GB+ free space for data and models

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/handover-rl.git
cd handover-rl

# Setup environment
./setup_env.sh all

# Activate virtual environment
source venv/bin/activate
```

### Run Training

```bash
# Level 1: Quick validation (2 hours, 100 episodes)
./quick_train.sh 1

# Level 3: Validation (10 hours, 500 episodes) - Recommended
./quick_train.sh 3

# Level 5: Full training (35 hours, 1700 episodes) - Publication quality
./quick_train.sh 5
```

**See [Quick Start Guide](docs/training/QUICKSTART.md) for details**

---

## 📁 Project Structure

```
handover-rl/
├── src/
│   ├── environments/              # Gymnasium environment
│   │   └── satellite_handover_env.py  ✅ Algorithm-agnostic
│   ├── agents/                    # RL algorithms
│   │   ├── dqn_agent_v2.py       ✅ DQN (current)
│   │   └── ...                    🚧 PPO, Double DQN, etc. (Phase 1-3)
│   ├── strategies/                # Rule-based comparison methods
│   │   └── ...                    🚧 A4, D2, heuristics (Phase 4)
│   ├── trainers/                  🚧 Training logic (refactoring)
│   └── utils/                     # Utilities
│       └── satellite_utils.py    ✅ Stage 4 pool loading
├── docs/                          # Documentation
│   ├── architecture/              # Architecture design
│   ├── training/                  # Training guides
│   ├── algorithms/                # Algorithm baselines & literature
│   └── development/               # Implementation plans
├── config/                        # Configuration files
│   ├── data_gen_config.yaml
│   └── training_config.yaml
├── train_online_rl.py            ✅ Current training script
└── README.md                      # This file
```

---

## 📊 System Architecture

### Current Implementation

```
Environment (Gymnasium) ✅
    ↓
DQN Agent ✅
    ↓
Online Training with Experience Replay ✅
    ↓
Multi-Level Training Strategy ✅
```

### Planned Refactoring

```
Unified Training Entry (train.py) 🚧
    ↓
Trainer Layer (Off-policy / On-policy) 🚧
    ↓
Agent Layer (DQN / PPO / A2C) 🚧
    ↓
Environment Layer (Gymnasium) ✅ Already done
```

**See [Architecture Refactor](docs/architecture/ARCHITECTURE_REFACTOR.md) for details**

---

## 🎓 Scientific Rigor

### Data Sources
- ✅ **Real TLE Data**: Space-Track.org (79 Starlink TLE files, 82 days coverage)
- ✅ **Official Physics Models**:
  - ITU-R P.676-13 (atmospheric attenuation)
  - 3GPP TS 38.214 (RSRP/RSRQ/SINR calculations)
  - 3GPP TS 38.331 (A3/A4/A5/D2 handover events)
- ✅ **No Simplified Algorithms**: All implementations follow official specifications
- ✅ **No Mock Data**: Only real physical calculations from orbit-engine

### Constellation Choice
- **Starlink-only** (101 satellites)
- **Rationale**: Cross-constellation handover (Starlink↔OneWeb) not realistic
  - Literature review: NO papers do cross-constellation handover
  - Commercial reality: Separate networks (like AT&T vs Verizon)

**See [Constellation Choice](docs/architecture/CONSTELLATION_CHOICE.md)**

---

## 🧪 Multi-Level Training Strategy

### Progressive Validation

| Level | Satellites | Episodes | Duration | Use Case |
|-------|-----------|----------|----------|----------|
| 0 | 10 | 10 | 10 min | Smoke test |
| 1 | 20 | 100 | 2 hours | Quick validation ⭐ Start here |
| 2 | 50 | 300 | 6 hours | Development |
| 3 | 101 | 500 | 10 hours | Validation (paper draft) |
| 4 | 101 | 1000 | 21 hours | Baseline (paper experiments) |
| 5 | 101 | 1700 | 35 hours | Full training (final experiments) |

**Rationale**: Avoid 35-hour training every iteration. Progressive validation enables faster development.

**See [Training Levels](docs/training/TRAINING_LEVELS.md)**

---

## 📚 Algorithm Support

### Project Scope (Baseline Framework)

**RL Baseline** (Phase 1):
- ✅ **DQN** (Deep Q-Network) - Standard RL baseline from literature

**Rule-based Baselines** (Phase 2):
- ✅ **Strongest RSRP** - Simple heuristic strategy
- ✅ **A4-based Strategy** - 3GPP A4 event + RSRP selection (validated for LEO)
- ✅ **D2-based Strategy** - 3GPP D2 event + distance selection (NTN-specific) ⭐

**Future Comparison**:
- ⭐ User's own algorithm vs above 4 baselines
- ❌ No need to implement other RL algorithms (D3QN, A2C, etc.)

### Literature Review Reference (Not in Scope)

For understanding LEO satellite handover research landscape, see [Baseline Algorithms](docs/algorithms/BASELINE_ALGORITHMS.md) which includes literature review of:
- D3QN, A2C, Rainbow DQN, SAC (有 handover 論文證據)
- PPO (用於 satellite scheduling，非 handover)

**Note**: These are for reference only, not in project implementation scope

---

## 📖 Documentation

**完整文檔索引**: [docs/README.md](docs/README.md) ⭐

### Current Status & Reports (2025-10-21) ⭐ NEW

| 文檔 | 說明 |
|------|------|
| **[Project Status](docs/PROJECT_STATUS.md)** | 當前項目狀態與待辦事項 ⭐ |
| **[BC 訓練總結](docs/reports/FINAL_SOLUTION_SUMMARY.md)** | 完整解決方案（必讀）|
| **[訓練報告 V4](docs/reports/TRAINING_REPORT_V4_FINAL.md)** | BC V4 訓練詳細報告 |
| **[數據洩漏診斷](docs/reports/DIAGNOSIS_100_ACCURACY.md)** | 100% 準確率問題分析 |
| **[閾值建議](docs/reports/FINAL_THRESHOLD_RECOMMENDATIONS.md)** | 數據驅動閾值設計 |
| **[清理報告](docs/reports/CLEANUP_REPORT.md)** | 項目結構整理記錄 |

### Quick References
- **[Quick Start](docs/training/QUICKSTART.md)** - Get started in 5 minutes
- **[Training Levels](docs/training/TRAINING_LEVELS.md)** - Multi-level strategy explained
- **[Gymnasium Migration](docs/training/GYMNASIUM_MIGRATION.md)** - Why Gymnasium, not gym

### Architecture & Design
- **[Architecture Refactor](docs/architecture/ARCHITECTURE_REFACTOR.md)** - Modular framework design
- **[Constellation Choice](docs/architecture/CONSTELLATION_CHOICE.md)** - Why Starlink-only
- **[Data Dependencies](docs/architecture/DATA_DEPENDENCIES.md)** - orbit-engine integration

### Algorithms & Research
- **[Baseline Algorithms](docs/algorithms/BASELINE_ALGORITHMS.md)** - Literature-backed algorithm selection
- **[Algorithm Guide](docs/algorithms/ALGORITHM_GUIDE.md)** - How to implement new algorithms
- **[Literature Review](docs/algorithms/LITERATURE_REVIEW.md)** - 2023-2025 papers summary

### Development
- **[Implementation Plan](docs/development/IMPLEMENTATION_PLAN.md)** - Phase-by-phase refactoring plan (修正版)
- **[Phase 2: Rule-based Methods](docs/development/PHASE2_RULE_BASED_METHODS.md)** - Detailed guide for implementing comparison methods ⭐

---

## 🔬 Research Contributions

### Novel Aspects
1. **Multi-Level Training Strategy**: 6 levels from 10min to 35hrs (progressive validation)
2. **Continuous Time Sampling**: Sliding window with configurable overlap
3. **Starlink-Specific**: 101 satellites from orbit-engine Stage 4 (real TLE data)
4. **Comprehensive Baseline Framework**: Modular architecture supporting RL and rule-based methods
5. **NTN-Specific Baselines**: First use of D2 event (3GPP Rel-17 NTN) as baseline ⭐

### Baseline Methods (For Algorithm Comparison)
**RL Baseline** (Phase 1):
- DQN (Deep Q-Network) - Standard RL baseline from literature

**Rule-based Baselines** (Phase 2):
- Strongest RSRP (Simple heuristic)
- A4-based Strategy (3GPP event + RSRP selection, validated for LEO)
- D2-based Strategy (3GPP Rel-17 event + distance selection, NTN-specific) ⭐

**Note**: A4/D2 are 3GPP measurement report triggers. Our baseline strategies supplement them with selection logic and handover decisions.

### Academic Compliance
- ✅ All parameters traceable to official sources
- ✅ No hardcoded values (satellite pool from Stage 4)
- ✅ No simplified algorithms
- ✅ Reproducible (seed-controlled)
- ✅ Rule-based methods use 3GPP standards + orbit-engine real data

---

## 📊 Performance Baselines

Based on literature (Graph RL, Frontiers 2023):
- **Handover Frequency**: 10-30% of timesteps (target)
- **Average RSRP**: > -95 dBm
- **Convergence**: ~1500-1700 episodes
- **Ping-Pong Rate**: < 10%

---

## 🛠️ Development Roadmap

### Phase 1: DQN Refactoring (Week 1-2)
- [ ] Create BaseAgent interface
- [ ] Implement OffPolicyTrainer
- [ ] Refactor DQN to new architecture
- [ ] **Preserve Multi-Level Training (P0 Critical)** ⭐
- [ ] Validation against current implementation

### Phase 2: Rule-based Baselines (Week 3)
- [ ] Implement 3 rule-based strategies (Strongest RSRP, A4-based, D2-based)
- [ ] Unified evaluation framework (RL + rule-based)
- [ ] Level 1 baseline evaluation (DQN + 3 rule-based)
- [ ] Baseline framework complete and ready for algorithm comparison

**Total Time**: 2-3 weeks

**目標**: 建立完整的 baseline 框架，包含 1 個 RL baseline (DQN) 和 3 個 rule-based baselines，作為未來算法對比的基礎
**See [Implementation Plan](docs/development/IMPLEMENTATION_PLAN.md)** | **[BASELINE_ALGORITHMS.md](docs/algorithms/BASELINE_ALGORITHMS.md)**

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@software{handover_rl_2025,
  title={Handover-RL: Modular RL Framework for LEO Satellite Handover},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/handover-rl}
}
```

---

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **orbit-engine**: https://github.com/yourusername/orbit-engine
- **Gymnasium**: https://gymnasium.farama.org/
- **Literature**: See [docs/algorithms/BASELINE_ALGORITHMS.md](docs/algorithms/BASELINE_ALGORITHMS.md)

---

**Status**: 🚧 Active Development - Phase 1 (DQN Refactoring)
**Version**: 1.0.0-dev
**Last Updated**: 2025-10-20 (建立 Baseline 框架)
**Estimated Completion**: 2-3 weeks (Phase 1-2)
**目標**: 建立包含 DQN + Rule-based baselines 的完整框架，作為未來算法對比的基礎
