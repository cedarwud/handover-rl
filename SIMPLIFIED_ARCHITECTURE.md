# 簡化架構方案：明確 orbit-engine 依賴

## 設計原則

**用戶洞察**：
> "如果最終都還是需要 orbit-engine 專案，那是否就不用這麼複雜了，
> 就是要在這個專案存在才能開始進行預處理跟訓練，問題會少一些"

**核心觀點**：✅ 正確！
- 不要過度設計 fallback 機制
- 明確職責分工：orbit-engine（數據）+ handover-rl（訓練）
- 簡化架構，減少維護負擔

---

## 簡化後的架構

### 項目職責劃分

```
orbit-engine/                    # 數據處理項目
├── Stage 1-4: TLE → 衛星池
├── Stage 5-6: 信號分析
└── 輸出: link_feasibility_output_*.json (29MB)
         ↓
         ↓ 明確依賴
         ↓
handover-rl/                    # RL 訓練項目
├── 讀取 orbit-engine Stage 4 輸出
├── 生成預計算表（可選，加速訓練）
└── DQN 訓練 + 評估
```

### 目錄結構（簡化版）

```
/home/sat/satellite/
├── orbit-engine/               # 必須存在（依賴項目）
│   ├── data/outputs/stage4/
│   │   └── link_feasibility_output_*.json
│   └── ...
│
├── handover-rl/                # 主項目
│   ├── data/
│   │   ├── satellite_pool/
│   │   │   ├── snapshot_v1.0.json      # 從 orbit-engine 快照（僅用於版本追蹤）
│   │   │   └── snapshot_v1.0.metadata.json
│   │   │
│   │   └── orbit_precompute/
│   │       └── orbit_precompute_30days.h5  # 本地生成（不追蹤）
│   │
│   ├── configs/config.yaml
│   ├── train_sb3.py
│   └── ...
```

---

## 數據管理策略（簡化版）

### 1. 運行時數據來源（唯一來源）

```python
# src/utils/satellite_utils.py (簡化版)

def load_satellites(constellation_filter='starlink'):
    """
    從 orbit-engine Stage 4 載入衛星池

    硬性要求：orbit-engine 必須存在於 sibling directory
    """
    stage4_dir = Path("/home/sat/satellite/orbit-engine/data/outputs/stage4")

    if not stage4_dir.exists():
        raise FileNotFoundError(
            "❌ orbit-engine Stage 4 輸出不存在\n"
            "\n"
            "handover-rl 依賴 orbit-engine 提供衛星池數據。\n"
            "\n"
            "設置步驟：\n"
            "  1. Clone orbit-engine 到 sibling directory:\n"
            "     cd /home/sat/satellite\n"
            "     git clone https://github.com/user/orbit-engine.git\n"
            "\n"
            "  2. 運行 orbit-engine Stage 4:\n"
            "     cd orbit-engine\n"
            "     ./run.sh --stage 4\n"
            "\n"
            "  3. 返回 handover-rl 繼續訓練:\n"
            "     cd ../handover-rl\n"
            "     python train_sb3.py --config configs/config.yaml\n"
        )

    # 載入 Stage 4 數據
    stage4_files = sorted(stage4_dir.glob("link_feasibility_output_*.json"))
    if not stage4_files:
        raise FileNotFoundError(
            "❌ 未找到 Stage 4 輸出文件\n"
            "請運行: cd ../orbit-engine && ./run.sh --stage 4"
        )

    latest_file = stage4_files[-1]

    with open(latest_file) as f:
        data = json.load(f)

    # 提取衛星池
    pools = data['pool_optimization']['optimized_pools']
    satellite_ids = [sat['satellite_id'] for sat in pools[constellation_filter]]

    # 提取元數據（用於新鮮度檢查）
    metadata = extract_metadata(data, latest_file)

    # 新鮮度檢查
    check_data_freshness(metadata, max_age_days=14)

    return satellite_ids, metadata
```

### 2. 快照數據（僅用於版本追蹤和論文可重現性）

**目的**：
- ✅ Git 追蹤訓練時使用的衛星池版本
- ✅ 論文可重現性（記錄使用的確切衛星池）
- ❌ 不用於實際訓練（訓練總是使用 orbit-engine 最新數據）

**生成方式**：
```bash
# 當完成一輪重要訓練後，創建快照
python tools/data/create_satellite_pool_snapshot.py \
  --orbit-engine-dir ../orbit-engine \
  --output data/satellite_pool/snapshot_v1.0.json \
  --version 1.0.0 \
  --description "Used for paper experiments (Dec 2025)"

# Git 追蹤快照
git add data/satellite_pool/snapshot_v1.0.json
git add data/satellite_pool/snapshot_v1.0.metadata.json
git commit -m "Add satellite pool snapshot v1.0.0 for paper reproducibility"
```

**快照內容**（30KB）：
```json
{
  "metadata": {
    "version": "1.0.0",
    "purpose": "Paper reproducibility snapshot",
    "source_file": "link_feasibility_output_20251126_074928.json",
    "tle_epoch": "2025-11-26",
    "snapshot_date": "2025-12-17T10:00:00Z",
    "data_age_days": 21,
    "orbit_engine_version": "4.0",
    "description": "Used for IEEE TAES paper experiments (Dec 2025)"
  },
  "satellite_pools": {
    "starlink": [
      {"satellite_id": "12345", "name": "STARLINK-1234", ...},
      // ... 101 satellites
    ]
  }
}
```

**用途**：
```latex
% 論文中引用
\subsection{Satellite Pool}
Training used 101 Starlink satellites selected by orbit-engine Stage 4
optimization (version 4.0, TLE epoch: 2025-11-26).
The exact satellite pool is preserved in the code repository
(\texttt{data/satellite\_pool/snapshot\_v1.0.json}) for reproducibility.
```

---

## 設置流程（簡化版）

### 新用戶設置

```bash
# Step 1: Clone 兩個項目到同一父目錄
cd /path/to/workspace
git clone https://github.com/user/orbit-engine.git
git clone https://github.com/user/handover-rl.git

# 目錄結構：
# /path/to/workspace/
# ├── orbit-engine/
# └── handover-rl/

# Step 2: 設置 orbit-engine
cd orbit-engine
./setup_env.sh
./run.sh --stage 4  # 生成衛星池（~10分鐘）

# Step 3: 設置 handover-rl
cd ../handover-rl
./setup_env.sh

# Step 4: 生成預計算表（可選，加速訓練）
python tools/generate_orbit_precompute.py \
  --duration-days 30 \
  --output data/orbit_precompute/precompute_30days.h5
# 時間：~30分鐘（一次性）

# Step 5: 訓練
python train_sb3.py --config configs/config.yaml --seed 42
# 速度：
# - 有預計算表：2500 episodes ~25分鐘
# - 無預計算表：2500 episodes ~20小時
```

### 自動化設置腳本

```bash
# setup_env.sh (增強版)

#!/bin/bash
set -e

echo "========================================="
echo "Handover-RL Environment Setup"
echo "========================================="
echo ""

# 1. 檢查 Python 版本
echo "✓ Checking Python version..."
python --version | grep -q "Python 3.1[0-9]" || {
    echo "❌ Python 3.10+ required"
    exit 1
}

# 2. 創建虛擬環境
echo "✓ Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# 3. 安裝依賴
echo "✓ Installing dependencies..."
pip install -r requirements.txt

# 4. 檢查 orbit-engine 依賴 ⭐ 關鍵檢查
echo ""
echo "========================================="
echo "Checking orbit-engine dependency..."
echo "========================================="

ORBIT_ENGINE_DIR="../orbit-engine"
STAGE4_DIR="$ORBIT_ENGINE_DIR/data/outputs/stage4"

if [ ! -d "$ORBIT_ENGINE_DIR" ]; then
    echo ""
    echo "❌ orbit-engine not found at $ORBIT_ENGINE_DIR"
    echo ""
    echo "handover-rl requires orbit-engine as a dependency."
    echo ""
    echo "Setup steps:"
    echo "  1. Clone orbit-engine to sibling directory:"
    echo "     cd .. && git clone https://github.com/user/orbit-engine.git"
    echo ""
    echo "  2. Run orbit-engine Stage 4:"
    echo "     cd orbit-engine && ./run.sh --stage 4"
    echo ""
    echo "  3. Re-run this setup script:"
    echo "     cd ../handover-rl && ./setup_env.sh"
    echo ""
    exit 1
fi

if [ ! -d "$STAGE4_DIR" ] || [ -z "$(ls -A $STAGE4_DIR/*.json 2>/dev/null)" ]; then
    echo ""
    echo "⚠️  orbit-engine found, but Stage 4 output missing"
    echo ""
    echo "Please run:"
    echo "  cd $ORBIT_ENGINE_DIR"
    echo "  ./run.sh --stage 4"
    echo ""
    echo "Then re-run: ./setup_env.sh"
    echo ""
    exit 1
fi

echo "✅ orbit-engine Stage 4 output found"
LATEST_STAGE4=$(ls -t $STAGE4_DIR/*.json | head -1)
echo "   Latest: $(basename $LATEST_STAGE4)"
echo ""

# 5. 檢查預計算表
echo "========================================="
echo "Checking orbit precompute table..."
echo "========================================="

PRECOMPUTE_FILE="data/orbit_precompute/orbit_precompute_30days.h5"

if [ -f "$PRECOMPUTE_FILE" ]; then
    FILE_SIZE=$(du -h "$PRECOMPUTE_FILE" | cut -f1)
    echo "✅ Precompute table found ($FILE_SIZE)"
    echo "   Training will use fast precompute mode (100-1000x speedup)"
else
    echo "⚠️  Precompute table not found"
    echo ""
    echo "Training will use real-time calculation (slow)."
    echo ""
    echo "To generate precompute table (~30 minutes):"
    echo "  python tools/generate_orbit_precompute.py \\"
    echo "    --duration-days 30 \\"
    echo "    --output data/orbit_precompute/precompute_30days.h5"
    echo ""
    read -p "Generate now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python tools/generate_orbit_precompute.py \
            --duration-days 30 \
            --output data/orbit_precompute/precompute_30days.h5 \
            --processes 16 \
            --yes
    fi
fi

echo ""
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "Quick Start:"
echo "  # Activate environment:"
echo "  source venv/bin/activate"
echo ""
echo "  # Run training:"
echo "  python train_sb3.py --config configs/config.yaml --seed 42"
echo ""
```

---

## README 簡化版

````markdown
# Handover-RL

Deep Reinforcement Learning for LEO Satellite Handover Optimization

## Prerequisites

**Required Dependency**: [orbit-engine](https://github.com/user/orbit-engine)

This project depends on `orbit-engine` for satellite pool data.
Both projects must be cloned to the same parent directory:

```
workspace/
├── orbit-engine/      ← Required dependency
└── handover-rl/       ← This project
```

## Quick Start

### 1. Clone Projects

```bash
cd /path/to/workspace
git clone https://github.com/user/orbit-engine.git
git clone https://github.com/user/handover-rl.git
```

### 2. Setup orbit-engine (Dependency)

```bash
cd orbit-engine
./setup_env.sh
./run.sh --stage 4  # Generate satellite pool (~10 min)
```

### 3. Setup handover-rl

```bash
cd ../handover-rl
./setup_env.sh  # Auto-checks orbit-engine dependency
```

### 4. Generate Precompute Table (Optional, Recommended)

```bash
# One-time generation (~30 minutes)
python tools/generate_orbit_precompute.py \
  --duration-days 30 \
  --output data/orbit_precompute/precompute_30days.h5
```

**Training Speed**:
- With precompute: 2500 episodes in ~25 minutes ✅
- Without precompute: 2500 episodes in ~20 hours ⚠️

### 5. Train

```bash
python train_sb3.py --config configs/config.yaml --seed 42
```

## Data Freshness Management

### Updating Satellite Pool

orbit-engine Stage 4 data should be updated every **2 weeks** to maintain TLE freshness:

```bash
# Update orbit-engine data
cd ../orbit-engine
./run.sh --stage 4

# Regenerate precompute table with fresh data
cd ../handover-rl
python tools/generate_orbit_precompute.py \
  --duration-days 30 \
  --output data/orbit_precompute/precompute_30days.h5
```

### Creating Snapshots for Reproducibility

After important training runs (e.g., for paper submission):

```bash
# Create versioned snapshot
python tools/data/create_satellite_pool_snapshot.py \
  --orbit-engine-dir ../orbit-engine \
  --output data/satellite_pool/snapshot_v1.0.json \
  --version 1.0.0 \
  --description "Used for IEEE TAES paper (Dec 2025)"

# Commit snapshot
git add data/satellite_pool/snapshot_v1.0.*
git commit -m "Add satellite pool snapshot v1.0.0 for paper reproducibility"
```

Snapshots (~30KB) are tracked in Git for reproducibility, but **runtime always uses latest orbit-engine data**.
````

---

## 數據新鮮度檢查（簡化版）

```python
# src/utils/data_freshness.py

def check_data_freshness(metadata: dict, max_age_days: int = 14):
    """
    檢查 orbit-engine Stage 4 數據新鮮度

    Args:
        metadata: 從 Stage 4 文件提取的元數據
        max_age_days: 最大允許年齡（默認14天）

    Raises:
        DataFreshnessError: 數據過舊
    """
    tle_epoch = datetime.fromisoformat(metadata['tle_epoch'])
    age_days = (datetime.now() - tle_epoch).days

    if age_days > max_age_days:
        raise DataFreshnessError(
            f"\n"
            f"{'='*80}\n"
            f"❌ orbit-engine 數據過舊\n"
            f"{'='*80}\n"
            f"\n"
            f"TLE Epoch:     {metadata['tle_epoch']}\n"
            f"Data Age:      {age_days} 天 (建議 < {max_age_days} 天)\n"
            f"Source File:   {metadata['source_file']}\n"
            f"\n"
            f"⚠️  影響：\n"
            f"  • 位置誤差: {estimate_position_error(age_days)} km\n"
            f"  • RVT 誤差: {estimate_rvt_error(age_days)} 秒\n"
            f"  • 訓練結果可能不可靠\n"
            f"\n"
            f"解決方案：\n"
            f"  1. 更新 orbit-engine Stage 4:\n"
            f"     cd ../orbit-engine && ./run.sh --stage 4\n"
            f"\n"
            f"  2. 重新生成預計算表:\n"
            f"     python tools/generate_orbit_precompute.py --duration-days 30\n"
            f"\n"
            f"  3. 或設置 --allow-stale-data（不推薦）\n"
            f"\n"
            f"{'='*80}\n"
        )

    # 顯示新鮮度信息
    freshness_level = get_freshness_level(age_days)
    print(f"✅ 數據新鮮度: {freshness_level} ({age_days} 天前)")
    print(f"   TLE Epoch: {metadata['tle_epoch']}")
    print(f"   Source: {metadata['source_file']}")
    print()
```

---

## 優點總結（簡化方案）

### ✅ 架構優點
1. **職責清晰**: orbit-engine（數據）+ handover-rl（訓練）
2. **依賴明確**: README 清楚說明需要 orbit-engine
3. **減少複雜性**: 無 fallback 機制，無內嵌數據同步問題
4. **易於維護**: 數據更新只需運行 `./run.sh --stage 4`

### ✅ 學術優點
1. **數據可追溯**: 總是使用 orbit-engine 最新數據
2. **版本控制**: 快照機制記錄論文使用的確切衛星池
3. **可重現性**: 快照 + 訓練元數據 → 完整可重現
4. **新鮮度保證**: 自動檢查 14 天閾值

### ✅ 用戶體驗
1. **設置明確**: 一次性設置兩個項目
2. **自動檢查**: setup_env.sh 自動驗證依賴
3. **清晰錯誤**: 缺少依賴時提供明確指引
4. **靈活訓練**: 有/無預計算表都能訓練

---

## 維護計劃

### 每 2 週（定期維護）
```bash
# 更新 orbit-engine 數據
cd orbit-engine
./run.sh --stage 4

# 重新生成預計算表
cd ../handover-rl
python tools/generate_orbit_precompute.py --duration-days 30
```

### 論文投稿前（創建快照）
```bash
# 創建快照
python tools/data/create_satellite_pool_snapshot.py \
  --version 1.0.0 \
  --description "IEEE TAES 2025 submission"

# Git 提交
git add data/satellite_pool/snapshot_v1.0.*
git commit -m "Add snapshot for IEEE TAES paper"
```

---

## 結論

**簡化後的架構**：
- ✅ 明確依賴 orbit-engine（不用假裝獨立）
- ✅ 簡化數據管理（無 fallback，無同步問題）
- ✅ 保持學術嚴謹（新鮮度檢查，版本快照）
- ✅ 減少維護負擔（一個數據源，清晰流程）

**Git 追蹤**：
- 30KB 快照（版本追蹤和論文可重現性）
- 不追蹤 2.6GB 預計算表（本地生成）
- 不追蹤 29MB Stage 4 JSON（由 orbit-engine 管理）
