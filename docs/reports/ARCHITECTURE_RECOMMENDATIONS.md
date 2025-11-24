# 專案架構優化建議

**分析日期**: 2024-11-24
**基於**: ARCHITECTURE_ANALYSIS.md 的深入分析

---

## ✅ 總體評估

### 當前結構評分: **9/10** (優秀)

**優點**:
- ✅ `src/` 內容完全正確 (可重用庫代碼)
- ✅ `scripts/` 內容完全正確 (獨立腳本)
- ✅ `train.py` 和 `evaluate.py` 位置正確 (根目錄)
- ✅ 符合 ML/Research 專案最佳實踐

**發現的問題**:
- ⚠️ 3 個資料夾可能重複或過時

---

## 🔍 發現的具體問題

### 問題 1: config/ 與 src/configs/ 可能混淆

**當前狀態**:
```
handover-rl/
├── config/                      ⚠️ 用戶配置 (YAML)
│   ├── diagnostic_config*.yaml     # 診斷配置
│   └── strategies/                 # Handover 策略配置
│       ├── a4_based.yaml
│       ├── d2_based.yaml
│       └── strongest_rsrp.yaml
│
└── src/
    └── configs/                 ✅ 訓練配置 (Python)
        └── training_levels.py      # Level 0-6 配置
```

**分析**:
- `config/`: 用戶配置文件 (YAML) ← **外部配置**
- `src/configs/`: 代碼配置 (Python) ← **內部配置**

**判斷**: ✅ **不重複，但命名容易混淆**

**建議**:

#### 選項 A: 重命名以避免混淆 (推薦)
```bash
# 將 config/ 重命名為 user_configs/
mv config/ user_configs/

# 或更明確的名稱
mv config/ yaml_configs/
mv config/ experiments/
```

#### 選項 B: 保持現狀，添加 README
```bash
# config/README.md
# User Configuration Files
This directory contains user-editable YAML configuration files.
- diagnostic_config.yaml: Diagnostic test configurations
- strategies/: Handover strategy configurations

# src/configs/README.md
# Internal Configuration Modules
This directory contains Python code for training level configurations.
Not meant to be edited by users directly.
```

#### 選項 C: 合併 (不推薦)
```bash
# 將所有配置移到 config/
config/
├── yaml/              # 用戶 YAML 配置
│   ├── diagnostic.yaml
│   └── strategies/
└── python/            # Python 代碼配置
    └── training_levels.py
```

**推薦**: **選項 A** (重命名為 `user_configs/` 或 `experiments/`)

---

### 問題 2: data/ 包含大型 HDF5 文件

**當前狀態**:
```
handover-rl/data/  (5.3 GB)
├── orbit_precompute_30days_optimized.h5  (2.3 GB)  ← 當前使用
├── orbit_precompute_30days_full.h5       (1.4 GB)  ← 舊版本
├── orbit_precompute_30days.h5            (1.4 GB)  ← 舊版本
├── orbit_precompute_7days.h5             (319 MB)  ← 測試用
├── orbit_precompute_1day_test.h5         (49 MB)   ← 測試用
└── training_metrics.csv                  (6.9 KB)
```

**問題**:
- ✅ 當前使用: `orbit_precompute_30days_optimized.h5`
- ❌ 舊版本: 3 個過時的 HDF5 文件 (3.1 GB)

**判斷**: ⚠️ **包含過時數據，可清理**

**建議**:

#### 清理舊版本 precompute 文件
```bash
# 1. 備份當前使用的文件
mkdir -p data/active
mv data/orbit_precompute_30days_optimized.h5 data/active/

# 2. 歸檔舊版本
mkdir -p archive/data/precompute-old
mv data/orbit_precompute_30days_full.h5 archive/data/precompute-old/
mv data/orbit_precompute_30days.h5 archive/data/precompute-old/

# 3. 保留測試文件 (或歸檔)
mkdir -p data/test
mv data/orbit_precompute_7days.h5 data/test/
mv data/orbit_precompute_1day_test.h5 data/test/

# 4. 移除 training_metrics.csv (應該在 output/ 中)
mv data/training_metrics.csv archive/data/
```

**效果**: 減少 3.1 GB 根目錄數據

---

### 問題 3: checkpoints/ 為空目錄

**當前狀態**:
```
handover-rl/checkpoints/  (空)
```

**判斷**: ❌ **無用空目錄**

**說明**:
- Checkpoints 實際上在 `output/level*/checkpoints/`
- 根目錄的 `checkpoints/` 可能是舊設計殘留

**建議**: 刪除空目錄
```bash
rmdir checkpoints/  # 如果確認為空
```

---

### 問題 4: logs/ 與 output/*/logs/ 可能重複

**當前狀態**:
```
handover-rl/logs/  (81 MB)
├── level0_*.log
├── level1_training.log
├── memory_diagnosis.log
├── batch_test.log
└── ...

handover-rl/output/
├── level0_*/logs/
├── level1_*/logs/
├── level5_*/logs/
└── level6_*/logs/
```

**判斷**: ⚠️ **部分重複，但有些是臨時測試日誌**

**分析**:
- `logs/`: 臨時測試、診斷日誌 (不屬於任何 level)
- `output/*/logs/`: 正式訓練日誌 (屬於特定 level)

**建議**:

#### 選項 A: 清理整理 (推薦)
```bash
# 1. 將屬於 output/ 的日誌移過去
mv logs/level0_*.log archive/logs/  # 已在 output/level0_*/logs/
mv logs/level1_training.log archive/logs/

# 2. 保留診斷和測試日誌
mkdir -p logs/diagnostics
mkdir -p logs/tests
mv logs/memory_diagnosis.log logs/diagnostics/
mv logs/batch_test.log logs/tests/

# 3. 舊日誌歸檔
mv logs/level0_smoke_test.log archive/logs/
```

#### 選項 B: 保持現狀
- `logs/`: 用於臨時日誌、診斷
- `output/*/logs/`: 用於正式訓練日誌

**推薦**: **選項 B** (保持現狀，清晰區分用途)

但添加 `logs/README.md`:
```markdown
# Temporary and Diagnostic Logs

This directory contains:
- Temporary test logs
- Diagnostic logs (memory profiling, debugging)
- Logs that don't belong to a specific training level

For training logs, see: `output/level*/logs/`
```

---

## 📋 src/ vs scripts/ 明確區分標準 (總結)

### 快速判斷流程圖

```
新增一個 Python 文件
        ↓
    [判斷問題]
        ↓
┌───────────────────────────────────┐
│ 1. 這個文件會被多處 import 嗎？    │
│    (≥2 個其他文件使用)             │
└───────────────────────────────────┘
         ↓YES              ↓NO
    ┌─────────┐      ┌─────────┐
    │ src/    │      │ 繼續... │
    └─────────┘      └─────────┘
                          ↓
┌───────────────────────────────────┐
│ 2. 主要用途是獨立執行嗎？          │
│    (python xxx.py)                 │
└───────────────────────────────────┘
         ↓YES              ↓NO
    ┌─────────┐      ┌─────────┐
    │ 繼續... │      │ src/    │
    └─────────┘      └─────────┘
         ↓
┌───────────────────────────────────┐
│ 3. 完成特定任務嗎？                │
│    (數據處理/批次訓練/生成圖表)    │
└───────────────────────────────────┘
         ↓YES              ↓NO
    ┌─────────┐      ┌─────────┐
    │ scripts/│      │ 根目錄  │
    └─────────┘      └─────────┘
                          ↓
                   (如果是主要入口)
```

---

### 具體範例

#### ✅ 放在 src/ 的範例

```python
# src/agents/dqn_agent.py
class DQNAgent(BaseAgent):
    """可重用的 DQN agent"""
    def select_action(self, state):
        ...

# 被使用:
# - train.py
# - evaluate.py
# - tests/test_agent_fix.py
```

```python
# src/utils/satellite_utils.py
def load_stage4_optimized_satellites():
    """可重用的工具函數"""
    ...

# 被使用:
# - train.py
# - evaluate.py
# - scripts/batch_train.py
```

---

#### ✅ 放在 scripts/ 的範例

```python
# scripts/generate_orbit_precompute.py
def main():
    """生成 precompute 表格 (執行一次)"""
    adapter = OrbitEngineAdapter(config)
    generator = OrbitPrecomputeGenerator(adapter)
    generator.generate(...)

if __name__ == "__main__":
    main()

# 使用: python scripts/generate_orbit_precompute.py
# 不被其他代碼導入
```

```python
# scripts/paper/plot_learning_curves.py
def main():
    """生成論文圖表 (完成特定任務)"""
    data = extract_training_data()
    plot_curves(data)
    save_figure()

if __name__ == "__main__":
    main()

# 使用: python scripts/paper/plot_learning_curves.py
# 不被其他代碼導入
```

---

#### ✅ 放在根目錄的範例

```python
# train.py (主要入口點)
def main():
    """訓練模型 - 用戶最常用的功能"""
    ...

if __name__ == "__main__":
    main()

# 使用: python train.py --level 5
# 最簡潔的命令
```

---

### 邊界案例處理

#### 案例 1: 即是函數又是腳本
```python
# 最佳實踐: 分離
# src/utils/reward_analyzer.py (可重用函數)
def analyze_rewards(log_file):
    """可被多處使用"""
    return statistics

# scripts/analyze_all_rewards.py (獨立腳本)
from utils.reward_analyzer import analyze_rewards

def main():
    """批次分析所有實驗"""
    for log in glob("output/*/logs/*.log"):
        stats = analyze_rewards(log)
        print(stats)

if __name__ == "__main__":
    main()
```

#### 案例 2: 只被一個文件使用
```python
# 如果只被 train.py 使用，但邏輯複雜
# → 還是放在 src/ (為了模組化)

# src/trainers/checkpoint_manager.py
class CheckpointManager:
    """管理 checkpoint 保存/載入"""
    # 雖然只被 train.py 使用
    # 但邏輯複雜，值得獨立成模組
    ...
```

---

## 🎯 推薦的清理行動

### 高優先級 (建議執行)

1. **重命名 config/ 為 user_configs/**
   ```bash
   mv config/ user_configs/
   # 避免與 src/configs/ 混淆
   ```

2. **清理 data/ 中的舊 precompute 文件**
   ```bash
   mkdir -p archive/data/precompute-old
   mv data/orbit_precompute_30days_full.h5 archive/data/precompute-old/
   mv data/orbit_precompute_30days.h5 archive/data/precompute-old/
   # 節省 3.1 GB 空間
   ```

3. **刪除空的 checkpoints/ 目錄**
   ```bash
   rmdir checkpoints/
   ```

---

### 中優先級 (可選)

4. **整理 logs/ 目錄**
   ```bash
   mkdir -p logs/diagnostics
   mkdir -p logs/tests
   mv logs/memory_diagnosis.log logs/diagnostics/
   mv logs/batch_test.log logs/tests/
   ```

5. **添加 README 文件**
   ```bash
   # 在各主要目錄添加 README.md
   # 說明目錄用途和內容
   ```

---

### 低優先級 (未來優化)

6. **考慮添加 notebooks/ 目錄**
   ```bash
   mkdir notebooks/
   # 用於 Jupyter notebook 分析
   ```

7. **考慮添加 setup.py**
   ```python
   # 如果需要 pip install -e .
   from setuptools import setup, find_packages

   setup(
       name="handover-rl",
       version="1.0.0",
       packages=find_packages(where="src"),
       package_dir={"": "src"},
       ...
   )
   ```

---

## 📊 清理後的預期結構

```
handover-rl/                     ✅ 優化後
│
├── train.py                     ✅ 主要入口 (訓練)
├── evaluate.py                  ✅ 主要入口 (評估)
│
├── src/                         ✅ 核心庫 (可重用代碼)
│   ├── adapters/                   # 數據適配器
│   ├── agents/                     # RL agents
│   ├── environments/               # Gym environments
│   ├── trainers/                   # 訓練邏輯
│   ├── configs/                    # 訓練配置 (Python)
│   └── utils/                      # 工具函數
│
├── scripts/                     ✅ 獨立腳本
│   ├── generate_orbit_precompute.py
│   ├── append_precompute_day.py
│   ├── batch_train.py
│   ├── extract_training_data.py
│   └── paper/                      # 論文生成
│
├── tests/                       ✅ 測試
│   └── scripts/
│
├── user_configs/                ✅ 重命名 (避免混淆)
│   ├── diagnostic_config.yaml
│   └── strategies/
│
├── data/                        ✅ 清理後 (只保留當前使用)
│   ├── active/
│   │   └── orbit_precompute_30days_optimized.h5
│   └── test/
│       ├── orbit_precompute_7days.h5
│       └── orbit_precompute_1day_test.h5
│
├── logs/                        ✅ 整理後
│   ├── diagnostics/
│   ├── tests/
│   └── README.md
│
├── output/                      ✅ 訓練輸出
├── evaluation/                  ✅ 評估結果
├── figures/                     ✅ 論文圖表
├── tables/                      ✅ 論文表格
├── archive/                     ✅ 歸檔
│   └── data/
│       └── precompute-old/      # 舊 precompute 文件
│
├── requirements.txt             ✅
└── README.md                    ✅
```

---

## ✅ 最終建議總結

### 當前架構評估: **優秀 (9/10)**

**優點**:
- ✅ src/ 和 scripts/ 完全符合最佳實踐
- ✅ train.py 和 evaluate.py 位置正確
- ✅ 符合研究專案慣例

**需要改進**:
1. ⚠️ 重命名 `config/` 避免混淆
2. ⚠️ 清理 `data/` 中的舊文件 (節省 3.1 GB)
3. ⚠️ 刪除空的 `checkpoints/` 目錄

### src/ vs scripts/ 黃金規則

```
✅ src/      → 可重用庫代碼 (類、函數、被多處導入)
✅ scripts/  → 獨立腳本 (完成特定任務、不被導入)
✅ 根目錄    → 主要入口點 (train.py, evaluate.py)
```

---

**分析完成**: 2024-11-24
**結論**: 當前結構**基本正確**，只需少量清理即可達到最佳狀態
