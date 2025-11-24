# Scripts 目錄深度清理報告

**清理日期**: 2024-11-24
**清理範圍**: `scripts/` 目錄及所有子目錄
**清理原則**: 移除重複、歸檔過時、整理分類、測試文件歸位

---

## 📊 清理統計總覽

| 類別 | 數量 | 處理方式 |
|------|------|----------|
| 測試文件 | 6 個 | 移至 `tests/scripts/` |
| 調試腳本 | 5 個 | 移至 `archive/debug-scripts/` |
| 可視化腳本 | 9 個 | 整理至 `scripts/visualization/` |
| 過時腳本 | 3 個 | 移至 `archive/scripts/` |
| 合併目錄 | 1 個 | `verification/` 合併至 `validation/` |
| 刪除空目錄 | 1 個 | `testing/` 目錄已刪除 |
| **總計處理** | **25 個文件** | **全部整理完成** |

---

## 📁 詳細處理記錄

### 1. 測試文件 → `tests/scripts/` (6 個)

✅ **移動至專案測試目錄**

從 `scripts/` 移動到 `tests/scripts/`:
- `test_agent_fix.py` - DQN Agent memory leak fix 測試
- `test_evaluation_framework.py` - 評估框架測試
- `test_handover_event_loader.py` - Handover 事件加載器測試
- `test_safety_mechanism.py` - 安全機制測試（Episode 520-525）
- `testing/train_quick_test.py` - 快速訓練測試
- `testing/run_pre_refactoring_tests.sh` - 重構前測試腳本

**理由**: 測試文件應該統一放在 `tests/` 目錄，不應該在 `scripts/` 中。

---

### 2. 調試腳本 → `archive/debug-scripts/` (5 個)

🗄️ **歸檔已解決的調試問題**

- `diagnose_episode522.py` - Episode 522 問題診斷（已解決）
- `diagnose_memory_leak.py` - Memory leak 診斷（已解決）
- `pinpoint_memory_leak.py` - Memory leak 精確定位（已解決）
- `monitor_memory_fix.sh` - Memory fix 監控腳本（已解決）
- `monitor_safety_test.sh` - 安全測試監控腳本（已解決）

**創建時間**: 2025-11-18 至 2025-11-19
**問題狀態**: ✅ 全部已解決
**理由**: 這些是臨時調試腳本，問題已修復，歸檔保留作為歷史記錄。

---

### 3. 可視化腳本 → `scripts/visualization/` (9 個)

📊 **新建目錄統一管理**

創建 `scripts/visualization/` 並移入：
- `plot_episode920_comparison.py` - Episode 920 對比圖
- `plot_handover_analysis.py` - Handover 分析圖
- `plot_learning_curves.py` - 學習曲線圖
- `generate_performance_table.py` - 性能表格生成
- `generate_live_html.py` - 實時 HTML 生成
- `realtime_dashboard.py` - 實時儀表板
- `realtime_tensorboard.py` - 實時 TensorBoard
- `paper_style.py` - 論文風格設置
- `extract_training_data.py` - 訓練數據提取

**理由**: 所有繪圖和可視化相關的腳本應該集中管理，便於論文圖表生成。

---

### 4. 過時腳本 → `archive/scripts/` (3 個)

🗄️ **歸檔不再使用的腳本**

- `demo_comparison.py` - Demo 比較腳本（早期 demo）
- `run_level1_comparison.sh` - Level 1 比較腳本（已改用新方法）
- `evaluate_strategies.py` - 舊的策略評估框架（已被 `evaluate.py` 取代）

**創建時間**: 2025-10-25
**理由**:
- `demo_comparison.py`: 早期 demo，現在不需要
- `run_level1_comparison.sh`: Level 1 已過時，現在使用 Level 6
- `evaluate_strategies.py`: 根目錄已有更新的 `evaluate.py`

---

### 5. 目錄合併與刪除

#### 合併: `verification/` → `validation/`

**原因**: 功能重疊，統一管理更清晰

**驗證目錄原有文件**:
- `check_adapter_fields.py`
- `verify_dependencies.py`
- `verify_orbit_adapter.py`
- `verify_refactoring.py`
- `verify_tle_data.py`

**驗證目錄原有文件**:
- `diagnose_visibility.py`
- `stage0_academic_compliance.py`
- `stage1_unit_tests.py`
- `validate_refactored_framework.py`
- `run_full_validation.sh`

**合併後** `validation/` 共有 10 個文件，統一管理所有驗證相關功能。

#### 刪除: `testing/` 目錄

**原因**: 所有測試文件已移至 `tests/scripts/`，目錄為空

---

## ✅ 保留在 scripts/ 根目錄的核心腳本

清理後，`scripts/` 根目錄只保留 **4 個核心腳本**：

```
scripts/
├── append_precompute_day.py        # 擴展 precompute table
├── batch_train.py                  # 批次訓練（Level 6）
├── generate_orbit_precompute.py    # 生成 precompute table
└── monitor_batch_training.sh       # 監控批次訓練
```

**用途**:
- `generate_orbit_precompute.py` - **核心功能**: 生成 30 天軌道預計算表
- `append_precompute_day.py` - **擴展功能**: 按需擴展 precompute table
- `batch_train.py` - **訓練功能**: Level 6 批次訓練腳本
- `monitor_batch_training.sh` - **監控功能**: 監控批次訓練進度

---

## 📂 清理後的目錄結構

```
scripts/
├── append_precompute_day.py        # ✅ 核心腳本
├── batch_train.py                  # ✅ 核心腳本
├── generate_orbit_precompute.py    # ✅ 核心腳本
├── monitor_batch_training.sh       # ✅ 核心腳本
│
├── analysis/                       # 🔍 分析工具
│   └── analyze_satellite_visibility.py
│
├── benchmarks/                     # 📊 基準測試
│   ├── baseline_benchmark.py
│   └── baseline_metrics.txt
│
├── maintenance/                    # 🔧 維護腳本
│   ├── clean_gym.sh
│   └── update_requirements.sh
│
├── setup/                          # ⚙️ 設置腳本
│   └── check_dependencies.sh
│
├── training/                       # 🎯 訓練腳本
│   ├── bc/
│   │   └── train_offline_bc_v4_candidate_pool.py
│   ├── online_rl/
│   │   └── train_online_rl.py
│   ├── train_advanced.py
│   └── README.md
│
├── validation/                     # ✅ 驗證腳本（合併後）
│   ├── check_adapter_fields.py
│   ├── diagnose_visibility.py
│   ├── run_full_validation.sh
│   ├── stage0_academic_compliance.py
│   ├── stage1_unit_tests.py
│   ├── validate_refactored_framework.py
│   ├── verify_dependencies.py
│   ├── verify_orbit_adapter.py
│   ├── verify_refactoring.py
│   └── verify_tle_data.py
│
├── visualization/                  # 📈 可視化腳本（新建）
│   ├── extract_training_data.py
│   ├── generate_live_html.py
│   ├── generate_performance_table.py
│   ├── paper_style.py
│   ├── plot_episode920_comparison.py
│   ├── plot_handover_analysis.py
│   ├── plot_learning_curves.py
│   ├── realtime_dashboard.py
│   └── realtime_tensorboard.py
│
└── archive/                        # 🗄️ 歷史歸檔
    ├── (舊的離線 RL、修復等腳本)
    └── (已存在的歸檔內容)
```

---

## 🎯 清理效果對比

### Before (清理前)
```bash
scripts/ 根目錄: 25 個文件
子目錄數量: 10 個（包含 verification/, testing/）
測試文件位置: 散落在 scripts/ 中
```

### After (清理後)
```bash
scripts/ 根目錄: 4 個核心腳本
子目錄數量: 8 個（合併 verification，刪除 testing）
測試文件位置: 統一在 tests/scripts/
新增專門目錄: visualization/ (9 個腳本)
```

**改善**:
- ✅ 根目錄腳本減少 **84%** (25 → 4)
- ✅ 測試文件歸位到 `tests/`
- ✅ 可視化腳本統一管理
- ✅ 驗證功能合併，避免重複
- ✅ 調試腳本歸檔，保持整潔

---

## 📋 各子目錄用途說明

### 核心功能目錄

| 目錄 | 用途 | 文件數 |
|------|------|--------|
| `analysis/` | 衛星可見性分析等 | 1 |
| `benchmarks/` | 性能基準測試 | 2 |
| `training/` | 各種訓練方法（BC, Online RL等） | 4 |
| `validation/` | 驗證與測試（合併後） | 10 |
| `visualization/` | 繪圖與可視化（新建） | 9 |

### 輔助功能目錄

| 目錄 | 用途 | 文件數 |
|------|------|--------|
| `maintenance/` | 環境維護腳本 | 2 |
| `setup/` | 依賴檢查等設置 | 1 |
| `archive/` | 歷史歸檔 | 多個 |

---

## 💡 使用建議

### 1. 訓練相關

**生成 Precompute Table**:
```bash
python scripts/generate_orbit_precompute.py
```

**批次訓練 Level 6**:
```bash
python scripts/batch_train.py --level 6 --episodes 4174 --batch-size 100
```

**監控訓練**:
```bash
bash scripts/monitor_batch_training.sh
```

---

### 2. 可視化相關

**生成學習曲線**:
```bash
python scripts/visualization/plot_learning_curves.py
```

**生成性能表格**:
```bash
python scripts/visualization/generate_performance_table.py
```

**實時儀表板**:
```bash
python scripts/visualization/realtime_dashboard.py
```

---

### 3. 驗證相關

**運行完整驗證**:
```bash
bash scripts/validation/run_full_validation.sh
```

**檢查學術合規性**:
```bash
python scripts/validation/stage0_academic_compliance.py
```

---

## 🗑️ 後續清理建議

### 可選：進一步精簡

1-2 個月後，如果確認不再需要，可以刪除：

```bash
# 刪除歸檔的調試腳本（問題已解決）
rm -rf archive/debug-scripts/

# 刪除舊的離線 RL 腳本（已不使用）
rm -rf scripts/archive/offline_rl/

# 刪除舊的測試（已有新測試）
rm -rf scripts/archive/old_tests/
```

---

### 維護建議

1. **測試文件規範**:
   - 新的測試文件一律放在 `tests/` 目錄
   - 不要在 `scripts/` 中創建 `test_*.py` 文件

2. **可視化腳本**:
   - 新的繪圖腳本放在 `scripts/visualization/`
   - 保持命名規範 `plot_*.py` 或 `generate_*.py`

3. **調試腳本**:
   - 臨時調試腳本以 `diagnose_*.py` 命名
   - 問題解決後立即歸檔到 `archive/debug-scripts/`

4. **定期清理**:
   - 每月檢查 `scripts/` 根目錄
   - 及時歸檔不再使用的腳本

---

## ✅ 驗證

清理完成後，請驗證核心功能：

```bash
# 1. 檢查核心腳本
python scripts/batch_train.py --help
python scripts/generate_orbit_precompute.py --help

# 2. 檢查測試文件已移動
ls tests/scripts/

# 3. 檢查可視化腳本
ls scripts/visualization/

# 4. 檢查驗證腳本（合併後）
ls scripts/validation/
```

---

## 🎉 總結

本次深度清理成功地：

- ✅ **精簡根目錄**: 從 25 個文件減少到 4 個核心腳本（減少 84%）
- ✅ **測試歸位**: 6 個測試文件移至 `tests/scripts/`
- ✅ **歸檔調試**: 5 個已解決的調試腳本歸檔
- ✅ **新建分類**: 創建 `visualization/` 目錄，集中管理 9 個繪圖腳本
- ✅ **合併重複**: 合併 `verification/` 至 `validation/`，統一管理驗證功能
- ✅ **刪除冗餘**: 刪除空的 `testing/` 目錄
- ✅ **清晰結構**: 每個子目錄職責明確，便於維護

**scripts/ 目錄現在結構清晰、職責明確、易於維護！**

---

**生成時間**: 2024-11-24
**報告位置**: `/home/sat/satellite/handover-rl/SCRIPTS_CLEANUP_REPORT_2024-11-24.md`
