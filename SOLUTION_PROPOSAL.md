# Handover-RL 數據管理方案

## 問題分析

**當前狀況**：
- 硬依賴 orbit-engine Stage 4 輸出 (29MB JSON)
- 無法 clone-and-run
- 數據更新依賴外部項目

**用戶擔憂**：
- Fallback 機制可能使用過舊數據
- 訓練結果可能因數據過舊而無效
- 不知道使用的是哪個數據源

## 解決方案：數據版本管理 + 透明度追蹤

### 架構設計

```
handover-rl/
├── data/
│   ├── satellite_pool/
│   │   ├── embedded_v1.0.json          # 內嵌數據（~5MB，從 Stage 4 提取）
│   │   ├── embedded_v1.0.metadata.json # 數據元數據
│   │   └── .gitkeep
│   └── orbit_precompute/
│       └── .gitkeep (本地生成，不追蹤)
│
├── src/utils/
│   ├── satellite_utils.py              # 數據載入（帶版本檢查）
│   └── data_version_manager.py         # 新增：數據版本管理
│
└── tools/data/
    ├── extract_satellite_pool.py       # 從 orbit-engine 提取輕量級數據
    └── update_embedded_data.py         # 更新內嵌數據
```

### 核心機制

#### 1. 數據元數據追蹤 (metadata.json)

```json
{
  "version": "1.0.0",
  "source": "orbit-engine",
  "stage4_file": "link_feasibility_output_20251126_074928.json",
  "extraction_date": "2025-12-17T10:30:00Z",
  "tle_epoch": "2025-11-26",
  "satellite_count": {
    "starlink": 101,
    "oneweb": 24,
    "total": 125
  },
  "data_hash": "sha256:abc123...",
  "orbit_engine_version": "4.0",
  "compatibility": {
    "min_handover_rl_version": "4.2",
    "max_age_days": {
      "recommended": 14,  // 推薦：2週內
      "acceptable": 30,   // 可接受：1個月內
      "warning": 60,      // 警告：2個月（精度下降）
      "critical": 90      // 嚴重：3個月（不建議使用）
    }
  }
}
```

#### 2. 數據載入優先級 + 驗證

```python
# src/utils/satellite_utils.py (重構版)

def load_satellites_with_version_check(
    constellation_filter: str = 'starlink',
    max_data_age_days: int = 90,
    strict_mode: bool = True
):
    """
    載入衛星數據，帶版本和新鮮度檢查

    數據來源優先級：
    1. orbit-engine Stage 4 (如果存在且可訪問)
    2. 內嵌數據 (embedded_v*.json)
    3. 錯誤退出（strict_mode=True）或警告（strict_mode=False）

    Args:
        constellation_filter: 星座篩選
        max_data_age_days: 數據最大允許年齡（天）
        strict_mode: 嚴格模式（數據過舊時報錯 vs 警告）

    Returns:
        (satellite_ids, metadata_info)
    """

    # 嘗試來源 1: orbit-engine Stage 4
    try:
        sat_ids, metadata = _load_from_orbit_engine(constellation_filter)
        print("✅ 使用數據來源: orbit-engine Stage 4 (最新)")
        return sat_ids, {
            'source': 'orbit-engine',
            'freshness': 'current',
            'age_days': 0
        }
    except FileNotFoundError:
        print("⚠️  orbit-engine Stage 4 不可用，嘗試內嵌數據...")

    # 嘗試來源 2: 內嵌數據
    embedded_data, embedded_meta = _load_embedded_data(constellation_filter)

    # 檢查數據新鮮度
    age_days = _calculate_data_age(embedded_meta)

    if age_days > max_data_age_days:
        warning_msg = f"""
        ⚠️  WARNING: 使用的衛星數據已過時！

        數據來源: 內嵌數據 v{embedded_meta['version']}
        數據年齡: {age_days} 天 (建議 < {max_data_age_days} 天)
        TLE Epoch: {embedded_meta['tle_epoch']}
        提取日期: {embedded_meta['extraction_date']}

        ⚠️  可能影響：
        - 軌道參數已過時（LEO 衛星軌道會漂移）
        - 訓練結果可能不反映當前星座狀態
        - 部分衛星可能已退役或重新定位

        建議動作：
        1. 更新 orbit-engine 並重新生成 Stage 4 數據
        2. 運行: python tools/data/update_embedded_data.py
        3. 或接受風險並繼續（設置 strict_mode=False）
        """

        if strict_mode:
            raise DataFreshnessError(warning_msg)
        else:
            print(warning_msg)
            print("\n⚠️  strict_mode=False，繼續使用過舊數據（風險自負）\n")

    print(f"✅ 使用數據來源: 內嵌數據 v{embedded_meta['version']} ({age_days} 天前)")

    return embedded_data, {
        'source': 'embedded',
        'version': embedded_meta['version'],
        'age_days': age_days,
        'freshness': 'current' if age_days <= 30 else 'stale'
    }
```

#### 3. 訓練時數據來源記錄

```python
# train_sb3.py (修改)

# 在訓練開始時
satellite_ids, data_info = load_satellites_with_version_check(
    constellation_filter='starlink',
    max_data_age_days=90,
    strict_mode=True  # 預設嚴格模式
)

# 記錄到訓練日誌
training_metadata = {
    'start_time': datetime.now().isoformat(),
    'data_source': data_info['source'],
    'data_version': data_info.get('version', 'N/A'),
    'data_age_days': data_info['age_days'],
    'data_freshness': data_info['freshness'],
    'satellite_count': len(satellite_ids),
    'constellation': config['environment']['constellation_filter'],
    'random_seed': seed
}

# 保存到 output/academic_seed42/metadata.json
with open(output_dir / 'metadata.json', 'w') as f:
    json.dump(training_metadata, f, indent=2)

# 終端顯示
print(f"""
{'='*80}
訓練配置摘要
{'='*80}
數據來源: {data_info['source']}
數據年齡: {data_info['age_days']} 天 ({data_info['freshness']})
衛星數量: {len(satellite_ids)} ({config['environment']['constellation_filter']})
隨機種子: {seed}
{'='*80}
""")
```

#### 4. 數據提取工具

```python
# tools/data/extract_satellite_pool.py

"""
從 orbit-engine Stage 4 提取輕量級衛星池數據

用途：
1. 創建內嵌數據（~5MB，包含在 git repo）
2. 定期更新內嵌數據（當 orbit-engine 更新時）

提取內容：
- 衛星 ID 列表（NORAD catalog number）
- 基本軌道參數（高度、傾角、週期）
- 星座分類（Starlink/OneWeb）
- TLE epoch 信息
- 提取時間戳

不包含（保持輕量）：
- 詳細的可見性時間窗
- 逐時間步的 RSRP 值
- 完整的信號分析結果
"""

def extract_lightweight_satellite_pool(
    stage4_file: Path,
    output_file: Path,
    version: str = "1.0.0"
):
    """提取輕量級衛星池數據"""

    with open(stage4_file) as f:
        stage4_data = json.load(f)

    # 提取優化池
    pools = stage4_data['pool_optimization']['optimized_pools']

    # 構建輕量級數據
    lightweight_data = {
        'starlink': [],
        'oneweb': []
    }

    for constellation in ['starlink', 'oneweb']:
        for sat in pools[constellation]:
            lightweight_data[constellation].append({
                'satellite_id': sat['satellite_id'],
                'name': sat['name'],
                'norad_id': sat['satellite_id'],
                # 基本軌道參數（如果需要）
                'altitude_km': sat.get('altitude_km', None),
                'inclination_deg': sat.get('inclination_deg', None)
            })

    # 元數據
    metadata = {
        'version': version,
        'source': 'orbit-engine',
        'stage4_file': stage4_file.name,
        'extraction_date': datetime.now().isoformat(),
        'tle_epoch': _extract_tle_epoch(stage4_data),
        'satellite_count': {
            'starlink': len(lightweight_data['starlink']),
            'oneweb': len(lightweight_data['oneweb']),
            'total': len(lightweight_data['starlink']) + len(lightweight_data['oneweb'])
        },
        'data_hash': _compute_hash(lightweight_data),
        'orbit_engine_version': '4.0',  # 從 stage4_data 提取
        'compatibility': {
            'min_handover_rl_version': '4.2',
            'max_age_days': 90
        }
    }

    # 保存數據
    output_data = {
        'metadata': metadata,
        'satellite_pools': lightweight_data
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    # 單獨保存元數據
    metadata_file = output_file.with_suffix('.metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ 提取完成:")
    print(f"   數據文件: {output_file} ({output_file.stat().st_size / 1024:.1f} KB)")
    print(f"   元數據:   {metadata_file}")
    print(f"   衛星數量: {metadata['satellite_count']['total']}")
    print(f"   數據版本: v{version}")
```

### 使用流程

#### 開發者工作流（有 orbit-engine）

```bash
# 1. 更新 orbit-engine Stage 4
cd /home/sat/satellite/orbit-engine
./run.sh --stage 4

# 2. 提取輕量級數據到 handover-rl
cd /home/sat/satellite/handover-rl
python tools/data/extract_satellite_pool.py \
  --stage4-dir ../orbit-engine/data/outputs/stage4 \
  --output data/satellite_pool/embedded_v1.1.json \
  --version 1.1.0

# 3. Git 提交（內嵌數據會被追蹤）
git add data/satellite_pool/embedded_v1.1.json
git add data/satellite_pool/embedded_v1.1.metadata.json
git commit -m "Update embedded satellite data to v1.1.0 (TLE epoch: 2025-12-17)"

# 4. 訓練（自動使用最新 orbit-engine 數據）
python train_sb3.py --config configs/config.yaml --seed 42
# → 使用 orbit-engine Stage 4 ✅
```

#### 新用戶工作流（無 orbit-engine）

```bash
# 1. Clone 項目
git clone https://github.com/user/handover-rl.git
cd handover-rl

# 2. 設置環境
./setup_env.sh

# 3. 直接訓練（使用內嵌數據）
python train_sb3.py --config configs/config.yaml --seed 42

# → 輸出：
# ⚠️  orbit-engine Stage 4 不可用，嘗試內嵌數據...
# ✅ 使用數據來源: 內嵌數據 v1.1.0 (15 天前)
#
# ================================================================================
# 訓練配置摘要
# ================================================================================
# 數據來源: embedded
# 數據年齡: 15 天 (current)
# 衛星數量: 101 (starlink)
# 隨機種子: 42
# ================================================================================

# 4. 檢查數據來源（訓練後）
cat output/academic_seed42/metadata.json
```

#### 數據更新警告示例

```bash
# 如果內嵌數據過舊（>90 天）
python train_sb3.py --config configs/config.yaml --seed 42

# → 輸出：
# ⚠️  orbit-engine Stage 4 不可用，嘗試內嵌數據...
#
# ⚠️  WARNING: 使用的衛星數據已過時！
#
# 數據來源: 內嵌數據 v1.0.0
# 數據年齡: 120 天 (建議 < 90 天)
# TLE Epoch: 2025-08-19
# 提取日期: 2025-08-20T10:30:00Z
#
# ⚠️  可能影響：
# - 軌道參數已過時（LEO 衛星軌道會漂移）
# - 訓練結果可能不反映當前星座狀態
# - 部分衛星可能已退役或重新定位
#
# 建議動作：
# 1. 更新 orbit-engine 並重新生成 Stage 4 數據
# 2. 運行: python tools/data/update_embedded_data.py
# 3. 或接受風險並繼續（設置 strict_mode=False）
#
# DataFreshnessError: 數據過舊，訓練終止（strict_mode=True）

# 如果接受風險
python train_sb3.py --config configs/config.yaml --seed 42 --allow-stale-data
# → 繼續訓練，但會在 metadata.json 記錄警告
```

### 數據文件大小估算

```
內嵌數據結構（embedded_v1.0.json）：
{
  "metadata": {...},              # ~1 KB
  "satellite_pools": {
    "starlink": [                # 101 satellites × ~200 bytes = ~20 KB
      {
        "satellite_id": "12345",
        "name": "STARLINK-1234",
        "altitude_km": 550,
        "inclination_deg": 53.0
      },
      ...
    ],
    "oneweb": [...]               # 24 satellites × ~200 bytes = ~5 KB
  }
}

預估總大小: ~30 KB (遠小於 GitHub 限制)
```

### 配置文件更新

```yaml
# configs/config.yaml

data:
  # 數據來源策略
  satellite_pool:
    preferred_source: "orbit-engine"  # orbit-engine | embedded | auto
    fallback_to_embedded: true
    strict_mode: true                  # 數據過舊時報錯（vs 警告）
    max_data_age_days: 90              # 最大允許數據年齡

  # 預計算表（保持本地生成）
  orbit_precompute:
    enabled: false
    table_path: "data/orbit_precompute/orbit_precompute_30days.h5"
    generation_required: true  # 提醒用戶需要生成
```

### README 更新

````markdown
## Quick Start

### Option 1: 使用內嵌數據（快速開始）

```bash
# Clone 項目
git clone https://github.com/user/handover-rl.git
cd handover-rl

# 設置環境
./setup_env.sh

# 直接訓練（使用內嵌數據）
python train_sb3.py --config configs/config.yaml --seed 42
```

**注意**: 內嵌數據可能不是最新的。查看訓練日誌中的數據年齡警告。

### Option 2: 使用最新數據（推薦用於研究）

```bash
# 1. 設置 orbit-engine（sibling 目錄）
cd /home/sat/satellite
git clone https://github.com/user/orbit-engine.git
cd orbit-engine
./run.sh --stage 4

# 2. 訓練（自動使用最新 orbit-engine 數據）
cd ../handover-rl
python train_sb3.py --config configs/config.yaml --seed 42
```

### 更新內嵌數據

```bash
# 當 orbit-engine 更新後，更新內嵌數據
python tools/data/extract_satellite_pool.py \
  --stage4-dir ../orbit-engine/data/outputs/stage4 \
  --output data/satellite_pool/embedded_v1.1.json \
  --version 1.1.0

# 提交更新
git add data/satellite_pool/embedded_v*.json
git commit -m "Update embedded data to v1.1.0"
```
````

## 優點總結

✅ **Clone-and-Run**: 新用戶無需 orbit-engine 即可訓練
✅ **數據透明度**: 明確顯示使用的數據來源和年齡
✅ **新鮮度保證**: 自動檢查數據是否過舊（可配置閾值）
✅ **可追溯性**: 每次訓練記錄數據來源到 metadata.json
✅ **靈活更新**: 開發者可輕鬆更新內嵌數據
✅ **輕量 Repo**: 內嵌數據僅 ~30 KB（vs 29 MB Stage 4 JSON）
✅ **向後兼容**: 優先使用 orbit-engine，fallback 到內嵌數據
✅ **風險提示**: 數據過舊時明確警告，避免無效訓練

## 實現計劃

1. **Phase 1: 數據提取工具**
   - 實現 `tools/data/extract_satellite_pool.py`
   - 從當前 Stage 4 生成 embedded_v1.0.json

2. **Phase 2: 載入邏輯重構**
   - 修改 `src/utils/satellite_utils.py`
   - 添加 `src/utils/data_version_manager.py`
   - 實現版本檢查和新鮮度驗證

3. **Phase 3: 訓練集成**
   - 修改 `train_sb3.py` 記錄數據元數據
   - 更新 README 和文檔

4. **Phase 4: 測試和驗證**
   - 測試所有數據來源場景
   - 驗證警告和錯誤處理
   - 多種子訓練測試

## 預期文件變更

```
新增文件：
+ data/satellite_pool/embedded_v1.0.json (~30 KB)
+ data/satellite_pool/embedded_v1.0.metadata.json (~1 KB)
+ tools/data/extract_satellite_pool.py
+ tools/data/update_embedded_data.py
+ src/utils/data_version_manager.py

修改文件：
M src/utils/satellite_utils.py (重構載入邏輯)
M train_sb3.py (添加元數據記錄)
M configs/config.yaml (添加數據策略配置)
M README.md (更新 Quick Start)
M .gitignore (追蹤 data/satellite_pool/*.json)
```
