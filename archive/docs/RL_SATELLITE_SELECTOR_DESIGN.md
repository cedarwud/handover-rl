# RL 訓練專用衛星選擇器設計文檔

**創建日期**: 2025-10-20
**目的**: 為 handover-rl 生成專門的衛星列表，取代 orbit-engine Stage 1-6 輸出

---

## 問題背景

### orbit-engine Stage 4 輸出不適合 RL 訓練

| 維度 | orbit-engine (Stage 1-6) | handover-rl 需求 | 差距 |
|------|-------------------------|------------------|------|
| **目的** | 前端 3D 渲染（Blender） | RL 換手策略訓練 | 目標不同 |
| **衛星數量** | 101 顆 | **800+ 顆** | 8x 差距 |
| **時間範圍** | 94 分鐘（單軌道週期） | **數天**（確保連續可見） | 30x+ 差距 |
| **篩選標準** | 單顆衛星個別可見品質 | **同時可見數量** ≥ 3-5 顆 | 標準不同 |
| **平均可見衛星** | 1.96 顆 | **4-6 顆** | 2-3x 差距 |
| **換手機會** | 0% handover rate ❌ | **持續換手機會** | 致命差距 |

**結論**: orbit-engine Stage 4 是為「單次完整軌道週期的前端渲染」優化的，**不適合 RL 訓練**。

---

## 設計目標

### 1. 核心目標
為 handover-rl 生成專用衛星列表，確保：
- ✅ 持續有 **3-5 顆衛星同時可見**
- ✅ 提供充足的 **換手機會**（A3/A4 事件）
- ✅ 軌道覆蓋 NTPU 上空
- ✅ 可見時間連續性（避免頻繁中斷）

### 2. 非目標
- ❌ 不需要考慮前端 3D 渲染需求
- ❌ 不受限於單軌道週期（94分鐘）
- ❌ 不需要最佳化單顆衛星的可見品質

---

## 技術規格

### 輸入數據

**TLE 數據源**:
```
/home/sat/satellite/tle_data/
├─ starlink/tle/          ← 80 個 TLE 文件（~5000+ 顆衛星）
└─ oneweb/tle/            ← 82 個 TLE 文件（~600+ 顆衛星）
```

**目標地點**:
```
NTPU (National Taipei University)
├─ 緯度: 25.0136°N
├─ 經度: 121.3676°E
└─ 海拔: 20m
```

**時間範圍**:
```
起始時間: 2025-10-18 (TLE epoch)
持續時間: 7 天（確保覆蓋多個軌道週期）
時間步長: 10 秒（RL 訓練時間解析度）
```

---

### 輸出數據

**衛星列表文件**: `handover-rl/data/selected_satellites.json`

```json
{
  "metadata": {
    "ground_station": {
      "name": "NTPU",
      "lat": 25.0136,
      "lon": 121.3676,
      "alt_m": 20
    },
    "time_range": {
      "start": "2025-10-18T00:00:00Z",
      "end": "2025-10-25T00:00:00Z",
      "duration_hours": 168
    },
    "selection_criteria": {
      "min_elevation_deg": 10,
      "min_rsrp_dbm": -100,
      "target_concurrent_visible": 5,
      "total_satellites_selected": 850
    },
    "coverage_stats": {
      "mean_concurrent_visible": 5.2,
      "min_concurrent_visible": 3,
      "max_concurrent_visible": 8,
      "handover_opportunity_rate": 0.45
    }
  },
  "satellites": [
    {
      "satellite_id": "48251",
      "constellation": "starlink",
      "tle_file": "starlink_2025-10-18.tle",
      "visibility_windows": [
        {
          "start": "2025-10-18T00:15:30Z",
          "end": "2025-10-18T00:24:15Z",
          "duration_sec": 525,
          "peak_elevation_deg": 45.2,
          "peak_rsrp_dbm": -35.5
        }
      ],
      "total_visible_duration_sec": 12500,
      "coverage_percentage": 7.4
    }
  ]
}
```

---

## 選擇算法

### 階段 1: 粗篩（Coarse Filtering）

**目標**: 從 5000+ 顆衛星快速篩選出可能覆蓋 NTPU 的衛星

**方法**: 軌道參數快速檢查
```python
def coarse_filter(tle_list, ground_station):
    """
    快速篩選：根據軌道傾角和週期判斷是否可能覆蓋地面站

    SOURCE: Orbital mechanics (Vallado 2013)
    - 軌道傾角 > 地面站緯度 → 可能覆蓋
    - LEO 軌道週期 90-120 分鐘
    """
    candidates = []

    for tle in tle_list:
        # 提取軌道參數（無需 SGP4 傳播）
        inclination = tle.inclination_deg
        period_min = calculate_period(tle.mean_motion)

        # 快速判斷
        if inclination >= ground_station.lat:  # 軌道傾角檢查
            if 90 <= period_min <= 120:  # LEO 軌道週期檢查
                candidates.append(tle)

    return candidates  # 預期: ~2000-3000 顆
```

**預期輸出**: ~2000-3000 顆候選衛星

---

### 階段 2: 可見性計算（Visibility Calculation）

**目標**: 計算每顆候選衛星在 7 天內的可見性窗口

**方法**: SGP4 傳播 + 仰角/距離計算
```python
def calculate_visibility_windows(satellite_id, tle, ground_station, time_range):
    """
    計算可見性窗口

    SOURCE: orbit-engine SGP4Calculator
    - 仰角 ≥ 10° → 可見
    - RSRP ≥ -100 dBm → 可連接
    """
    visibility_windows = []

    current_time = time_range.start
    in_window = False
    window_start = None

    while current_time <= time_range.end:
        # 使用 orbit-engine 的 SGP4Calculator
        position = sgp4_calculator.propagate(tle, current_time)
        elevation = calculate_elevation(position, ground_station)
        rsrp = calculate_rsrp(position, ground_station)  # 3GPP + ITU-R

        is_visible = (elevation >= 10.0) and (rsrp >= -100.0)

        if is_visible and not in_window:
            # 可見窗口開始
            window_start = current_time
            in_window = True
        elif not is_visible and in_window:
            # 可見窗口結束
            window_end = current_time
            visibility_windows.append({
                'start': window_start,
                'end': window_end,
                'duration_sec': (window_end - window_start).total_seconds()
            })
            in_window = False

        current_time += timedelta(seconds=10)  # 10秒步長

    return visibility_windows
```

**預期輸出**: 每顆衛星的可見性窗口列表

---

### 階段 3: 覆蓋率分析（Coverage Analysis）

**目標**: 計算時間軸上每個時間點的同時可見衛星數量

**方法**: 時間軸掃描
```python
def analyze_concurrent_visibility(satellites_visibility, time_range):
    """
    分析同時可見衛星數量

    目標: 確保大部分時間有 3-5 顆衛星同時可見
    """
    # 創建時間軸（7天，10秒步長）
    timeline = []
    current_time = time_range.start

    while current_time <= time_range.end:
        # 統計此時刻可見的衛星數量
        visible_sats = []

        for sat in satellites_visibility:
            for window in sat['visibility_windows']:
                if window['start'] <= current_time <= window['end']:
                    visible_sats.append(sat['satellite_id'])
                    break

        timeline.append({
            'time': current_time,
            'visible_count': len(visible_sats),
            'visible_satellites': visible_sats
        })

        current_time += timedelta(seconds=10)

    # 統計覆蓋率
    visible_counts = [t['visible_count'] for t in timeline]

    coverage_stats = {
        'mean': np.mean(visible_counts),
        'min': np.min(visible_counts),
        'max': np.max(visible_counts),
        'percentile_25': np.percentile(visible_counts, 25),
        'percentile_75': np.percentile(visible_counts, 75)
    }

    return timeline, coverage_stats
```

**預期輸出**:
```
覆蓋率統計:
├─ 平均同時可見: 5.2 顆
├─ 最少同時可見: 3 顆
├─ 最多同時可見: 8 顆
├─ 25th percentile: 4 顆
└─ 75th percentile: 6 顆
```

---

### 階段 4: 智能選擇（Intelligent Selection）

**目標**: 從候選衛星中選出最優的 800+ 顆

**方法**: 多目標優化
```python
def intelligent_selection(satellites_visibility, timeline, target_count=850):
    """
    智能選擇衛星

    優化目標:
    1. 最大化覆蓋率（時間覆蓋百分比）
    2. 最小化覆蓋缺口（連續無衛星時間）
    3. 平衡同時可見數量（避免過多或過少）
    4. 優先選擇高 RSRP 衛星
    """

    # 1. 計算每顆衛星的價值分數
    satellite_scores = []

    for sat in satellites_visibility:
        # 覆蓋時間百分比
        total_visible_sec = sum([w['duration_sec'] for w in sat['visibility_windows']])
        coverage_pct = total_visible_sec / (7 * 24 * 3600) * 100

        # 可見窗口數量（越多越好，表示經常出現）
        window_count = len(sat['visibility_windows'])

        # 平均 RSRP（越高越好）
        avg_rsrp = np.mean([w.get('peak_rsrp_dbm', -100) for w in sat['visibility_windows']])

        # 綜合分數
        score = (
            coverage_pct * 0.4 +      # 覆蓋率權重 40%
            window_count * 0.3 +      # 窗口數量權重 30%
            (avg_rsrp + 100) * 0.3    # RSRP 權重 30% (歸一化到 [0, 100])
        )

        satellite_scores.append({
            'satellite_id': sat['satellite_id'],
            'score': score,
            'coverage_pct': coverage_pct,
            'window_count': window_count,
            'avg_rsrp': avg_rsrp
        })

    # 2. 按分數排序
    satellite_scores.sort(key=lambda x: x['score'], reverse=True)

    # 3. 選擇前 N 顆
    selected = satellite_scores[:target_count]

    return selected
```

**預期輸出**: 850 顆最優衛星列表

---

### 階段 5: 驗證與優化（Validation & Optimization）

**目標**: 驗證選擇的衛星是否滿足 RL 訓練需求

**驗證指標**:
```python
def validate_selection(selected_satellites, timeline):
    """
    驗證選擇結果

    必須滿足的條件:
    1. 平均同時可見 ≥ 4 顆
    2. 最少同時可見 ≥ 2 顆（避免完全中斷）
    3. 換手機會率 ≥ 30%（時間點上有 A3/A4 事件的百分比）
    4. 覆蓋缺口 ≤ 5 分鐘（最長連續無衛星時間）
    """

    # 重新計算覆蓋率（僅使用選中的衛星）
    timeline_selected = filter_timeline(timeline, selected_satellites)

    visible_counts = [t['visible_count'] for t in timeline_selected]

    validation_results = {
        'mean_concurrent_visible': np.mean(visible_counts),
        'min_concurrent_visible': np.min(visible_counts),
        'max_concurrent_visible': np.max(visible_counts),

        # 換手機會率：有 ≥2 顆可見衛星的時間百分比
        'handover_opportunity_rate': sum([1 for c in visible_counts if c >= 2]) / len(visible_counts),

        # 最長覆蓋缺口
        'max_gap_minutes': calculate_max_gap(timeline_selected),

        # 通過/失敗
        'pass': True  # 由各項指標決定
    }

    # 檢查是否通過
    if validation_results['mean_concurrent_visible'] < 4.0:
        validation_results['pass'] = False
        validation_results['reason'] = 'Mean concurrent visible < 4'

    if validation_results['min_concurrent_visible'] < 2:
        validation_results['pass'] = False
        validation_results['reason'] = 'Min concurrent visible < 2'

    if validation_results['handover_opportunity_rate'] < 0.3:
        validation_results['pass'] = False
        validation_results['reason'] = 'Handover opportunity rate < 30%'

    return validation_results
```

---

## 實現計劃

### 腳本架構

```
handover-rl/
├─ scripts/
│  └─ data_generation/
│     ├─ rl_satellite_selector.py       ← 主腳本
│     ├─ coverage_analyzer.py           ← 覆蓋率分析工具
│     └─ satellite_scorer.py            ← 衛星評分工具
│
└─ data/
   ├─ selected_satellites.json          ← 選中的衛星列表（輸出）
   ├─ coverage_timeline.json            ← 覆蓋率時間軸（診斷用）
   └─ selection_report.md               ← 選擇報告（可視化）
```

### 依賴項

**使用 orbit-engine 算法**:
```python
# 已有的 adapter 可以直接使用
from src.adapters.orbit_engine_adapter import OrbitEngineAdapter

# 共享算法:
- SGP4Calculator          ← 軌道傳播
- GPPTS38214SignalCalculator ← RSRP 計算
- ITURPhysicsCalculator   ← 路徑損耗
```

**使用共享 TLE 數據**:
```python
import os
from pathlib import Path

# 從環境變量讀取 TLE 路徑
tle_dir = os.getenv('SATELLITE_TLE_DATA_DIR', '../tle_data')
tle_path = Path(tle_dir)

starlink_tle_dir = tle_path / 'starlink' / 'tle'
oneweb_tle_dir = tle_path / 'oneweb' / 'tle'
```

---

## 時間估算

| 階段 | 任務 | 預估時間 | 說明 |
|------|------|---------|------|
| 1 | 粗篩（2000-3000顆） | 1 分鐘 | 僅讀取 TLE 參數 |
| 2 | 可見性計算（2000顆 × 7天） | **30-60 分鐘** | SGP4 傳播最耗時 |
| 3 | 覆蓋率分析 | 5 分鐘 | 時間軸掃描 |
| 4 | 智能選擇 | 1 分鐘 | 評分排序 |
| 5 | 驗證優化 | 2 分鐘 | 指標計算 |
| **總計** | | **40-70 分鐘** | 一次性運行 |

**優化建議**:
- 可以使用多進程加速 SGP4 傳播（Python multiprocessing）
- 可以緩存中間結果（可見性窗口）

---

## 成功標準

選擇成功的標準:
- ✅ 選出 **800-900 顆衛星**
- ✅ **平均同時可見 ≥ 4 顆**
- ✅ **最少同時可見 ≥ 2 顆**
- ✅ **換手機會率 ≥ 30%**
- ✅ **最長覆蓋缺口 ≤ 5 分鐘**

如果不滿足:
1. 增加衛星數量（例如 1000 顆）
2. 延長時間範圍（例如 14 天）
3. 降低仰角門檻（例如 5°）

---

## 與 orbit-engine 的對比

| 項目 | orbit-engine Stage 4 | RL Satellite Selector |
|------|---------------------|----------------------|
| **選擇數量** | 101 顆 | 850 顆 |
| **時間範圍** | 94 分鐘 | 7 天 |
| **優化目標** | 單顆可見品質 | 同時可見數量 |
| **平均可見** | 1.96 顆 | 5+ 顆 |
| **換手機會** | 0% | 30-50% |
| **適用場景** | 前端 3D 渲染 | RL 訓練 |

---

**設計版本**: 1.0
**文檔日期**: 2025-10-20
**下一步**: 實現 `rl_satellite_selector.py` 腳本
