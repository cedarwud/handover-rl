# Scripts 目錄最終分析報告

**分析日期**: 2024-11-24 03:15
**分析範圍**: `scripts/` 完整目錄結構
**發現**: ❌ 仍有重複和問題需要解決

---

## 🚨 發現的問題

### 問題 1: scripts/paper/ 的依賴損壞

**嚴重程度**: ❌ **CRITICAL**

`scripts/paper/` 的 4 個腳本都依賴 `extract_training_data.py`，但這個文件已被移到歸檔：

```python
# scripts/paper/plot_learning_curves.py (Line 35)
from scripts.extract_training_data import extract_episode_data

# scripts/paper/plot_handover_analysis.py (Line 26)
from scripts.extract_training_data import extract_episode_data

# scripts/paper/generate_performance_table.py (Line 29)
from scripts.extract_training_data import extract_episode_data, compute_statistics
```

**位置問題**:
- ❌ 依賴文件: `archive/scripts-obsolete/visualization/extract_training_data.py`
- ✅ 可能的替代: `tools/extract_training_metrics.py`

**結果**: `scripts/paper/` 的所有腳本無法運行！

---

### 問題 2: scripts/archive/ 冗余

**嚴重程度**: ⚠️ **MEDIUM**

`scripts/archive/` 包含 14 個舊文件（總計 ~80KB），這些文件：

1. **offline_rl/** (3 files)
   - `offline_rl_train_dqn.py` (4.0K)
   - `offline_rl_evaluate.py` (3.5K)
   - `offline_rl/generate_dataset.py` (3.5K)
   - `offline_rl/data_generation/generate_1day_125sats.py` (4.1K)

   **狀態**: 項目不使用 Offline RL，只使用 DQN

2. **old_tests/** (5 files)
   - `test_dynamic_pool_selection.py` (3.7K)
   - `test_environment.py` (6.0K)
   - `test_targeted_generation.py` (4.8K)
   - `test_satellite_visibility.py` (2.9K)
   - `test_fresh_generation.py` (1.2K)

   **狀態**: 舊的測試，現在有 `tests/` 目錄

3. **fixes/** (3 files)
   - `fix_hardcoding.py` (3.5K)
   - `verify_data_generation_fix.py` (8.8K)
   - `verify_placeholder_fix.py` (7.9K)

   **狀態**: 一次性修復腳本，問題已解決

4. **test_end_to_end.py** (16K)
   **狀態**: 舊的端到端測試

**問題**: 為什麼 `scripts/` 內還有 `archive/`？歸檔文件應該在根目錄的 `archive/`

---

### 問題 3: 功能重複

**嚴重程度**: ⚠️ **MEDIUM**

兩個數據提取工具功能重疊：

1. **archive/scripts-obsolete/visualization/extract_training_data.py**
   - 從訓練日誌（.log 文件）提取數據
   - 使用正則表達式解析日誌
   - 輸出: DataFrame with episode, reward, loss, handovers

2. **tools/extract_training_metrics.py**
   - 從 TensorBoard 事件文件提取數據
   - 使用 TensorBoard API
   - 輸出: JSON 統計數據

**差異**: 數據源不同（.log vs TensorBoard events），但目的相同

---

## 📊 當前 scripts/ 結構

```
scripts/                              (總計: 20 個文件)
├── 核心腳本 (4 個) ✅
│   ├── batch_train.py               (8.4K) - Level 6 批次訓練
│   ├── generate_orbit_precompute.py  (8.3K) - 生成 precompute table
│   ├── append_precompute_day.py      (8.9K) - 擴展 precompute table
│   └── monitor_batch_training.sh     (1.2K) - 監控批次訓練
│
├── paper/ (4 個) ❌ 無法運行
│   ├── plot_learning_curves.py      (14K) - 學習曲線圖
│   ├── plot_handover_analysis.py    (14K) - Handover 分析圖
│   ├── generate_performance_table.py (9.8K) - 性能表格
│   └── paper_style.py               (11K) - 論文風格設置
│
└── archive/ (12 個) ⚠️ 應移出 scripts/
    ├── offline_rl_train_dqn.py      (4.0K)
    ├── offline_rl_evaluate.py       (3.5K)
    ├── test_end_to_end.py           (16K)
    ├── offline_rl/
    │   ├── generate_dataset.py      (3.5K)
    │   └── data_generation/generate_1day_125sats.py (4.1K)
    ├── old_tests/ (5 files, 18.6K total)
    └── fixes/ (3 files, 20.2K total)
```

---

## 🎯 建議的解決方案

### 方案 A: 激進清理（推薦）

**目標**: scripts/ 只保留真正需要且可運行的文件

#### 1. 修復 paper/ 腳本的依賴

**選項 1**: 移動 `extract_training_data.py` 到正確位置
```bash
# 從歸檔移回
mv archive/scripts-obsolete/visualization/extract_training_data.py scripts/

# 或移到 tools/
mv archive/scripts-obsolete/visualization/extract_training_data.py tools/
```

**選項 2**: 刪除 paper/ 腳本（如果論文已完成或不需要）
```bash
rm -rf scripts/paper/
```

**選項 3**: 修改 paper/ 腳本使用 `tools/extract_training_metrics.py`
- 需要重構代碼，工作量大

#### 2. 移動 scripts/archive/ 到根目錄

```bash
# 移動所有 scripts/archive/ 內容到根目錄 archive/
mv scripts/archive/* archive/scripts/
rm -rf scripts/archive/
```

**理由**:
- 歸檔文件不應該在 scripts/ 中
- 應該集中管理在根目錄 `archive/`

#### 3. 最終結構（激進方案）

```
scripts/
├── batch_train.py                  # ✅ Level 6 批次訓練
├── generate_orbit_precompute.py    # ✅ 生成 precompute table
├── append_precompute_day.py        # ✅ 擴展 precompute table
├── monitor_batch_training.sh       # ✅ 監控批次訓練
├── extract_training_data.py        # ✅ 數據提取（從歸檔移回）
│
└── paper/                          # ✅ 論文圖表（修復後可用）
    ├── plot_learning_curves.py
    ├── plot_handover_analysis.py
    ├── generate_performance_table.py
    └── paper_style.py

總計: 9 個文件（從 20 個減少到 9 個，減少 55%）
```

---

### 方案 B: 保守方案

保留 paper/ 但標記為 "需要修復"

```
scripts/
├── batch_train.py                  # ✅ 可用
├── generate_orbit_precompute.py    # ✅ 可用
├── append_precompute_day.py        # ✅ 可用
├── monitor_batch_training.sh       # ✅ 可用
│
└── paper/                          # ⚠️ 需要修復依賴
    ├── README.md                   # 新增：說明依賴問題
    ├── plot_learning_curves.py
    ├── plot_handover_analysis.py
    ├── generate_performance_table.py
    └── paper_style.py
```

並將 `scripts/archive/` 移到根目錄 `archive/scripts/`

---

## 🔍 功能重複分析

### extract_training_data.py vs extract_training_metrics.py

| 特性 | extract_training_data.py | extract_training_metrics.py |
|------|-------------------------|----------------------------|
| **位置** | archive/scripts-obsolete/visualization/ | tools/ |
| **數據源** | 訓練日誌 (.log 文件) | TensorBoard 事件文件 |
| **解析方式** | 正則表達式 | TensorBoard API |
| **輸出格式** | pandas DataFrame | JSON + 打印統計 |
| **功能** | 提供 `extract_episode_data()` 函數 | 獨立腳本，無法作為模組導入 |
| **依賴性** | paper/ 腳本依賴它 | 獨立使用 |

**結論**:
- ❌ 不能直接替換，因為功能不完全相同
- ✅ 如果需要 paper/ 腳本，必須保留 `extract_training_data.py`
- ⚠️ 如果論文已完成，可以刪除整個 paper/ 目錄

---

## 📋 行動清單

### 必須執行（修復損壞的依賴）

- [ ] **選擇方案**: 激進清理 vs 保守方案

  **如果選擇激進清理**:
  - [ ] 移動 `extract_training_data.py` 回 `scripts/`
  - [ ] 測試 paper/ 腳本能否運行
  - [ ] 移動 `scripts/archive/` 到 `archive/scripts/`
  - [ ] 更新文檔

  **如果選擇保守方案**:
  - [ ] 移動 `scripts/archive/` 到 `archive/scripts/`
  - [ ] 在 `scripts/paper/README.md` 註明依賴問題
  - [ ] 標記 paper/ 為 "需要修復"

### 可選執行（進一步優化）

- [ ] 合併 `extract_training_data.py` 和 `extract_training_metrics.py` 功能
- [ ] 評估 paper/ 腳本是否還需要（論文狀態）
- [ ] 清理 `archive/scripts-obsolete/` 中不再需要的文件

---

## 🎯 推薦方案

**推薦: 方案 A（激進清理）**

理由：
1. **修復依賴**: paper/ 腳本需要 `extract_training_data.py`
2. **結構清晰**: 歸檔文件不應該在 scripts/ 中
3. **極簡化**: 從 20 個文件減少到 9 個核心文件
4. **可維護**: 每個文件都有明確用途且可正常運行

執行步驟：
```bash
# 1. 移回必要的依賴
mv archive/scripts-obsolete/visualization/extract_training_data.py scripts/

# 2. 移動 scripts/archive/ 到根目錄
mkdir -p archive/scripts-old/
mv scripts/archive/* archive/scripts-old/
rm -rf scripts/archive/

# 3. 測試 paper/ 腳本
python scripts/paper/plot_learning_curves.py --help
```

---

## ✅ 驗證清單

完成清理後，驗證：

```bash
# 1. 檢查核心腳本
python scripts/batch_train.py --help
python scripts/generate_orbit_precompute.py --help

# 2. 檢查 paper/ 腳本
python scripts/paper/plot_learning_curves.py --help
python scripts/paper/plot_handover_analysis.py --help

# 3. 確認 scripts/ 結構
ls scripts/
ls scripts/paper/

# 4. 確認沒有 scripts/archive/
test ! -d scripts/archive && echo "✅ scripts/archive/ 已移除"
```

---

**生成時間**: 2024-11-24 03:15
**報告位置**: `/home/sat/satellite/handover-rl/SCRIPTS_FINAL_ANALYSIS.md`
