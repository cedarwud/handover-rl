# 最終深度清理方案

**基於完整代碼分析**
**日期**: 2025-11-08
**分析深度**: 文件級別代碼引用追蹤

---

## 📋 分析結果總結

### ✅ 確認使用中的文件

#### src/ (核心代碼)
```
✅ src/agents/dqn/              # train.py 使用
✅ src/agents/baseline/         # evaluate.py 使用
✅ src/agents/base_agent.py
✅ src/agents/replay_buffer.py
✅ src/environments/
✅ src/trainers/
✅ src/adapters/
✅ src/configs/
✅ src/utils/
```

#### config/
```
✅ diagnostic_config.yaml        # 最終訓練使用（Nov 3）
```

#### data/
```
✅ training_metrics.csv          # 訓練指標
```

---

## ❌ 確認應歸檔/刪除的文件

### src/ 清理

#### 1. src/agents/dqn_agent_v2.py + dqn_network.py
**發現**: 被舊的 `train_online_rl.py` 和測試使用
**決定**: 歸檔到 `archive/src/agents/`

#### 2. src/strategies/ (整個目錄)
**發現**:
- 被 `scripts/evaluate_strategies.py`, `demo_comparison.py` 等使用
- 已被 `src/agents/baseline/` 取代
- 只有舊的評估腳本在用

**決定**: 歸檔到 `archive/src/strategies/`

#### 3. src/models/bc_policy.py
**發現**: BC (Behavior Cloning) 相關
**決定**: 歸檔到 `archive/src/models/`

#### 4. src/archive/ (整個目錄)
**發現**: 完整的 offline_rl 舊系統（7個文件）
**決定**: 移動到 `archive/src/offline_rl/`

#### 5. src/environments/satellite_handover_env.py.backup_single_obj
**決定**: 刪除備份文件

---

### config/ 清理

#### 1. 歸檔舊配置
```
❌ conservative_dqn_config.yaml          → archive/config/
❌ epsilon_fixed_config.yaml             → archive/config/
❌ data_gen_config.yaml                  → archive/config/
❌ training_config.yaml                  → archive/config/
```

#### 2. 刪除備份
```
🗑️ data_gen_config.yaml.backup
🗑️ data_gen_config.yaml.backup_before_stability_fix
```

#### 3. config/archive/ 目錄
**決定**: 內容移動到主 `archive/config/historical/`

#### 4. config/strategies/ 目錄
**檢查後決定**: 如果是配置文件，保留；如果空，刪除

---

### data/ 清理

#### 1. rl_training_dataset_temporal.h5 (852KB)
**發現**: HDF5 文件，無代碼引用
**推測**: BC/offline RL 數據
**決定**: 歸檔到 `archive/data/`

---

### 統一 Archive 結構

**當前問題**: 5 個分散的 archive 目錄
```
./archive/          # 主歸檔
./src/archive/      # 源代碼歸檔
./config/archive/   # 配置歸檔
./scripts/archive/  # 腳本歸檔
./tests/archive/    # 測試歸檔
```

**統一方案**:
```
archive/
├── src/
│   ├── agents/
│   │   ├── dqn_agent_v1.py
│   │   ├── dqn_agent_v2.py
│   │   └── dqn_network.py
│   ├── strategies/               # 整個目錄
│   ├── models/
│   │   └── bc_policy.py
│   └── offline_rl/               # 從 src/archive/ 移來
│       ├── data_generation/
│       ├── rl_core/
│       └── handover_env.py
│
├── config/
│   ├── historical/               # 從 config/archive/ 移來
│   ├── conservative_dqn_config.yaml
│   ├── epsilon_fixed_config.yaml
│   ├── data_gen_config.yaml
│   └── training_config.yaml
│
├── data/
│   └── rl_training_dataset_temporal.h5
│
├── scripts/                      # 從 scripts/archive/ 移來
│   ├── offline_rl/
│   ├── old_tests/
│   └── fixes/
│
├── tests/                        # 從 tests/archive/ 移來
│   ├── test_end_to_end_offline_rl.py
│   └── test_integration_offline_rl.py
│
├── logs/                         # 已有
├── output/                       # 已有
├── evaluation/                   # 已有
└── docs/                         # 已有
```

---

### docs/ 簡化

#### 歸檔到 archive/docs/
```
❌ docs/algorithms/               # 可能過時
❌ docs/architecture/             # 可能過時
❌ docs/development/              # 可能過時
❌ docs/training/                 # 可能過時
❌ docs/validation/               # 可能過時
❌ docs/PROJECT_STATUS.md         # Oct 25，可能過時
❌ docs/PRE_REFACTORING_TESTS_COVERAGE.md
❌ docs/CLEANUP_HISTORY.md
❌ docs/RL_SATELLITE_SELECTOR_DESIGN.md
```

#### 保留在 docs/
```
✅ docs/ACADEMIC_ACCELERATION_PLAN.md
✅ docs/PAPER_FIGURES_GUIDE.md
✅ docs/INTEGRATION_GUIDE.md
✅ docs/README.md
✅ docs/reports/                  # BC 訓練報告
```

---

### scripts/ 簡化

#### 檢查 validation/ vs verification/
```bash
# 需要檢查內容是否重複
scripts/validation/ (5 files)
scripts/verification/ (5 files)
```

#### 可能的整合
```
scripts/
├── training/                     # ✅ 保留
├── plotting/                     # 🆕 整合 plot_*.py + paper_style.py
├── monitoring/                   # 🆕 整合 realtime_*.py + extract_*.py
├── archive/                      # ✅ 保留
└── (其他根據檢查結果決定)
```

---

### figures/ 優化

#### 刪除 PNG（只保留 PDF）
```
🗑️ *.png  (減少 ~1MB)
✅ *.pdf  (論文使用)
```

---

### checkpoints/ 整合

**當前問題**:
- `checkpoints/bc_v4_*/` - BC checkpoint
- `output/level5_20min_final/checkpoints/` - DQN checkpoint

**決定**:
- BC checkpoint → `archive/checkpoints/bc/`
- 統一使用 `output/<experiment>/checkpoints/` 結構

---

### 清理其他

#### 1. 刪除所有 __pycache__/ (12個)
```bash
find . -type d -name "__pycache__" -not -path "./venv/*" -exec rm -rf {} +
```

#### 2. 更新 .gitignore
```
__pycache__/
*.pyc
*.pyo
```

#### 3. API/Frontend
**決定**: 保留（API 運行中，Frontend 是有效組件）

---

## 🎯 執行計畫

### 階段 1: 統一 Archive (最重要)

```bash
# 1. 確保主 archive 結構存在
mkdir -p archive/{src/{agents,strategies,models,offline_rl},config/{historical},data,scripts,tests}

# 2. 移動 src/archive/ 內容
mv src/archive/offline_rl archive/src/
mv src/archive/dqn_agent_v1.py archive/src/agents/

# 3. 移動其他需要歸檔的 src 文件
mv src/agents/dqn_agent_v2.py archive/src/agents/
mv src/agents/dqn_network.py archive/src/agents/  # 檢查後確認
mv src/strategies archive/src/
mv src/models archive/src/

# 4. 移動 config/archive/ 內容
mv config/archive/* archive/config/historical/

# 5. 移動舊配置
mv config/{conservative_dqn_config.yaml,epsilon_fixed_config.yaml,data_gen_config.yaml,training_config.yaml} archive/config/

# 6. 移動 scripts/archive/ 內容（已經在 archive 中）
# 保持不動

# 7. 移動 tests/archive/ 內容
mv tests/archive/* archive/tests/

# 8. 清理空的 archive 目錄
rm -rf src/archive config/archive tests/archive
```

### 階段 2: 清理配置和數據

```bash
# 刪除 config 備份
rm config/data_gen_config.yaml.backup*

# 刪除 src 備份
rm src/environments/satellite_handover_env.py.backup_single_obj

# 歸檔 data
mv data/rl_training_dataset_temporal.h5 archive/data/
```

### 階段 3: 簡化 docs 和 scripts

```bash
# 歸檔 docs 子目錄
mv docs/{algorithms,architecture,development,training,validation} archive/docs/
mv docs/{PROJECT_STATUS.md,PRE_REFACTORING_TESTS_COVERAGE.md,CLEANUP_HISTORY.md,RL_SATELLITE_SELECTOR_DESIGN.md} archive/docs/

# scripts 整合（根據內容檢查後執行）
# TBD
```

### 階段 4: 優化 figures 和 checkpoints

```bash
# 刪除 PNG
rm figures/*.png

# 歸檔 BC checkpoint
mv checkpoints/bc_v4_* archive/checkpoints/bc/
```

### 階段 5: 清理緩存

```bash
# 刪除 __pycache__
find . -type d -name "__pycache__" -not -path "./venv/*" -exec rm -rf {} +
```

---

## 📊 預期效果

| 項目 | 清理前 | 清理後 | 改善 |
|------|--------|--------|------|
| Archive 目錄數 | 5個 | 1個 | ✅ 統一 |
| src/ 文件數 | 40個 | ~25個 | ✅ 37% 減少 |
| config/ 文件數 | 7個 + 子目錄 | 1個 | ✅ 86% 減少 |
| docs/ 子目錄 | 6個 | 1個 | ✅ 83% 減少 |
| 備份文件 | 3個 | 0個 | ✅ 清除 |
| __pycache__ | 12個 | 0個 | ✅ 清除 |
| figures/ | PDF+PNG | 只PDF | ✅ 減半 |

---

## ✅ 最終結構

```
handover-rl/
├── src/                        # ~25個核心文件
│   ├── agents/
│   │   ├── dqn/               # ✅ 當前使用
│   │   ├── baseline/          # ✅ 評估使用
│   │   ├── base_agent.py
│   │   └── replay_buffer.py
│   ├── environments/
│   ├── trainers/
│   ├── adapters/
│   ├── configs/
│   └── utils/
│
├── config/
│   └── diagnostic_config.yaml # ✅ 唯一配置
│
├── data/
│   └── training_metrics.csv   # ✅ 當前指標
│
├── docs/
│   ├── ACADEMIC_ACCELERATION_PLAN.md
│   ├── PAPER_FIGURES_GUIDE.md
│   ├── INTEGRATION_GUIDE.md
│   ├── README.md
│   └── reports/               # BC 報告
│
├── scripts/                    # 簡化後
│   ├── training/
│   ├── plotting/              # 🆕 整合
│   ├── monitoring/            # 🆕 整合
│   └── archive/
│
├── figures/
│   └── *.pdf                  # 只保留 PDF
│
├── checkpoints/               # 空（使用 output/<exp>/checkpoints/）
│
├── archive/                   # ✅ 統一歸檔
│   ├── src/
│   ├── config/
│   ├── data/
│   ├── scripts/
│   ├── tests/
│   ├── checkpoints/
│   ├── logs/
│   ├── output/
│   ├── evaluation/
│   └── docs/
│
└── (其他目錄保持)
```

---

## 🚦 執行確認

在執行前需要確認：

1. ✅ 已完成代碼引用分析
2. ✅ 已識別當前使用的文件
3. ✅ 已識別可歸檔的文件
4. ⚠️ **需要用戶最終確認後執行**

---

**準備就緒**: 等待用戶確認後開始執行深度清理
