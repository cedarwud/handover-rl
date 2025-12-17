# 數據架構澄清：衛星池 vs 預計算表

## 關鍵混淆點

**問題**：14天新鮮度要求是否意味著預計算表只能覆蓋14天？
**答案**：NO！這是兩個獨立的數據文件，有不同的新鮮度要求。

---

## 兩種數據文件

### 數據 1: 衛星池數據（Satellite Pool）

```
文件：data/satellite_pool/embedded_v1.0.json
大小：~30 KB
內容：衛星 ID 列表 + 基本元數據
```

**包含內容**：
```json
{
  "starlink": [
    {
      "satellite_id": "12345",
      "name": "STARLINK-1234",
      "altitude_km": 550,
      "inclination_deg": 53.0
    },
    // ... 101 satellites
  ],
  "metadata": {
    "tle_epoch": "2025-11-26",  // TLE 測量日期
    "extraction_date": "2025-12-01"
  }
}
```

**用途**：
- 告訴系統「有哪些衛星可用」
- 提供基本軌道參數（高度、傾角）
- 來源：orbit-engine Stage 4 優化池

**新鮮度要求**：14天內（TLE epoch）
- 原因：決定「衛星池組成」
- 衛星可能退役、重新定位、新增
- 超過14天，池組成可能不準確

**Git 追蹤**：YES（小文件，30KB）

---

### 數據 2: 預計算軌道表（Orbit Precompute Table）

```
文件：data/orbit_precompute/orbit_precompute_30days.h5
大小：~2.6 GB
內容：30天 × 518,400 時間步 × 102 衛星的詳細軌道狀態
```

**包含內容（HDF5 表）**：
```
時間步 0 (2025-12-01 00:00:00):
  衛星 12345:
    position: [x, y, z]
    velocity: [vx, vy, vz]
    elevation: 45.2°
    azimuth: 123.4°
    distance: 850 km
    rsrp: -85.3 dBm
    is_visible: True
    ...
  衛星 12346: ...
  ...

時間步 1 (2025-12-01 00:00:05):
  ...

... (518,400 時間步，5秒間隔，30天)
```

**用途**：
- 訓練時快速查詢（O(1)）：給定時間 → 衛星狀態
- 替代實時計算（100-1000x 加速）
- 包含完整物理計算結果（ITU-R, 3GPP, SGP4）

**覆蓋時間範圍**：30天（未來預測）
- 例如：2025-12-01 到 2025-12-31

**Git 追蹤**：NO（太大，2.6GB，本地生成）

---

## 關鍵區別：TLE Age vs 預測時長

### TLE Age（新鮮度）
```
TLE Epoch (測量日期):     2025-11-26
今天 (使用日期):          2025-12-10
TLE Age:                  14 天 ← 新鮮度指標
```

### 預測時長（覆蓋範圍）
```
預計算開始時間:  2025-12-10 00:00:00
預計算結束時間:  2026-01-09 23:59:55
預測時長:       30 天 ← 覆蓋範圍
```

### 總誤差時長 = TLE Age + 預測時長

這是關鍵！

**場景 1：TLE 很新，預測很遠**
```
TLE Epoch:       2025-12-08 (2天前，很新 ✅)
預計算開始:      2025-12-10
預計算結束:      2026-01-09 (30天後)

在預測末期（2026-01-09）：
  總誤差時長 = 2天(TLE age) + 30天(預測) = 32天
  預估位置誤差: ~10 km ⚠️
```

**場景 2：TLE 較舊，預測較近**
```
TLE Epoch:       2025-11-26 (14天前，邊界 ⚠️)
預計算開始:      2025-12-10
預計算結束:      2025-12-24 (14天後)

在預測末期（2025-12-24）：
  總誤差時長 = 14天(TLE age) + 14天(預測) = 28天
  預估位置誤差: ~8 km ⚠️
```

**場景 3：TLE 很舊，預測很遠（最糟）**
```
TLE Epoch:       2025-11-10 (30天前，很舊 🔴)
預計算開始:      2025-12-10
預計算結束:      2026-01-09 (30天後)

在預測末期（2026-01-09）：
  總誤差時長 = 30天(TLE age) + 30天(預測) = 60天
  預估位置誤差: ~30 km 🔴 不可靠！
```

---

## 預計算表的新鮮度策略

### 生成時機（Generation Time）

**最佳實踐**：
1. 使用**最新 TLE**（0-7天內）
2. 預計算**未來 14-30 天**
3. 在預計算**到期前重新生成**

**示例時間線**：
```
Day 0 (2025-12-01):
  • 獲取最新 TLE (epoch: 2025-11-30, age: 1天)
  • 生成預計算表 (2025-12-01 to 2025-12-31, 30天)
  • 總誤差時長: 1 + 30 = 31天 ✅

Day 15 (2025-12-16):
  • 仍使用同一預計算表
  • TLE age: 16天
  • 剩餘覆蓋: 15天
  • 總誤差時長 (at end): 16 + 15 = 31天 ✅

Day 30 (2025-12-31):
  • 預計算表到期
  • 需要重新生成 ⚠️
  • 獲取新 TLE (epoch: 2025-12-30, age: 1天)
  • 生成新表 (2026-01-01 to 2026-01-31, 30天)
```

### 預計算表元數據

```json
{
  "generation_info": {
    "tle_epoch": "2025-11-30",
    "generation_date": "2025-12-01T00:00:00Z",
    "tle_age_at_generation_days": 1,

    "coverage": {
      "start_time": "2025-12-01T00:00:00Z",
      "end_time": "2025-12-31T23:59:55Z",
      "duration_days": 30,
      "timestep_seconds": 5,
      "total_timesteps": 518400
    },

    "freshness": {
      "max_tle_age_at_end": 31,  // 1 + 30
      "estimated_error_at_start": "< 1 km",
      "estimated_error_at_end": "10-15 km",
      "recommended_regenerate_before": "2025-12-31"
    }
  }
}
```

---

## 實際數據管理策略

### 策略 A：滾動更新（推薦用於長期訓練）

```bash
# 每 2 週重新生成預計算表

Week 0 (2025-12-01):
  python tools/generate_orbit_precompute.py \
    --start-time "2025-12-01 00:00:00" \
    --end-time "2025-12-15 23:59:59" \
    --duration-days 14 \
    --output data/orbit_precompute/precompute_w1.h5

Week 2 (2025-12-15):
  python tools/generate_orbit_precompute.py \
    --start-time "2025-12-15 00:00:00" \
    --end-time "2025-12-29 23:59:59" \
    --duration-days 14 \
    --output data/orbit_precompute/precompute_w2.h5

Week 4 (2025-12-29):
  python tools/generate_orbit_precompute.py \
    --start-time "2025-12-29 00:00:00" \
    --end-time "2026-01-12 23:59:59" \
    --duration-days 14 \
    --output data/orbit_precompute/precompute_w3.h5
```

**優點**：
- TLE age 始終 < 1 天
- 總誤差時長: 1 + 14 = 15 天 ✅
- 高精度訓練

**缺點**：
- 需要定期重新生成（每2週，~30分鐘）

---

### 策略 B：一次性生成（推薦用於短期實驗）

```bash
# 一次生成 30 天，用完為止

python tools/generate_orbit_precompute.py \
  --start-time "2025-12-01 00:00:00" \
  --end-time "2025-12-31 23:59:59" \
  --duration-days 30 \
  --output data/orbit_precompute/precompute_30days.h5
```

**使用限制**：
- Day 0-15: 總誤差 1-16 天 ✅ 高精度
- Day 16-25: 總誤差 17-26 天 ⚠️ 可接受
- Day 26-30: 總誤差 27-31 天 ⚠️ 精度下降

**建議**：
- 在 Day 25 前完成所有訓練
- 或在 Day 15 重新生成新表

---

### 策略 C：Clone-and-Run Fallback（內嵌數據）

這是您原本關心的場景。

#### 內嵌數據包含什麼？

**Option 1：僅衛星池（30KB）**
```
data/satellite_pool/embedded_v1.0.json
- 衛星 ID 列表
- 基本軌道參數
- 不包含預計算表
```

**訓練時**：
```python
# 使用實時計算（無預計算表）
config = {
  'precompute': {
    'enabled': False  # 回退到 orbit-engine 實時計算
  }
}

# 訓練速度：慢（2500 episodes ~20 小時）
# 但仍然可用！
```

**Option 2：內嵌小型預計算表（600MB）**
```
data/orbit_precompute/embedded_7days.h5
- 7 天覆蓋範圍（vs 30天）
- 2.6GB → 600MB（可接受 Git LFS）
- 足夠運行 ~1000 episodes 測試
```

**Git LFS 可行性**：
- 600MB：可行（GitHub LFS 免費額度 1GB/月）
- 2.6GB：不可行（超出免費額度）

---

## 回答您的問題

### Q: "14天新鮮度是否意味著只能做14天的訓練？"

**A: NO！** 有幾種方案：

#### 方案 1：使用實時計算（無預計算表）
```
內嵌數據：衛星池（30KB）
訓練模式：orbit-engine 實時計算
訓練速度：慢（2500 episodes ~20 小時）
數據新鮮度：依賴 orbit-engine TLE
```

**Clone-and-Run**：
```bash
git clone https://github.com/user/handover-rl.git
./setup_env.sh
# 需要 orbit-engine（sibling directory）
python train_sb3.py --config configs/config.yaml --seed 42
# 使用實時計算，慢但可用
```

#### 方案 2：內嵌 7 天小型預計算表（Git LFS）
```
內嵌數據：
  - 衛星池（30KB）
  - 7天預計算表（600MB，Git LFS）

訓練模式：快速訓練（100x 加速）
訓練容量：~1000 episodes（足夠快速測試）
```

**Clone-and-Run**：
```bash
git clone https://github.com/user/handover-rl.git  # 自動下載 LFS
./setup_env.sh
python train_sb3.py --config configs/config.yaml --seed 42
# 快速訓練，但僅 7 天數據
```

#### 方案 3：本地生成 30 天預計算表（當前策略）
```
內嵌數據：衛星池（30KB）
訓練前：生成 2.6GB 預計算表（~30 分鐘）
訓練模式：完整快速訓練
```

**Setup**：
```bash
git clone https://github.com/user/handover-rl.git
./setup_env.sh

# 生成預計算表（一次性，30分鐘）
python tools/generate_orbit_precompute.py \
  --duration-days 30 \
  --output data/orbit_precompute/precompute_30days.h5

# 訓練（快速，2500 episodes ~25 分鐘）
python train_sb3.py --config configs/config.yaml --seed 42
```

---

## 推薦方案總結

| 方案 | 內嵌數據大小 | 需要 Git LFS | 訓練速度 | Clone-and-Run | 完整訓練能力 |
|------|------------|-------------|---------|--------------|-------------|
| **A: 僅衛星池** | 30 KB | ❌ | 慢 (20h) | ⚠️ 需要 orbit-engine | ✅ |
| **B: 7天預計算** | 600 MB | ✅ | 快 (1h) | ✅ | ⚠️ ~1000 episodes |
| **C: 本地生成** | 30 KB | ❌ | 快 (25m) | ⚠️ 需生成 (30m) | ✅ |

### 最終推薦：混合策略

```
handover-rl/
├── data/
│   ├── satellite_pool/
│   │   └── embedded_v1.0.json              # 30KB (Git 追蹤)
│   │
│   └── orbit_precompute/
│       ├── embedded_7days.h5               # 600MB (Git LFS，可選)
│       └── orbit_precompute_30days.h5      # 2.6GB (本地生成)
```

**數據載入優先級**：
```python
1. 檢查本地 30天預計算表 → 如果存在，使用（最快）
2. 檢查 Git LFS 7天預計算表 → 如果存在，使用（快速測試）
3. 檢查 orbit-engine → 如果存在，使用實時計算（慢但可用）
4. 提示用戶生成預計算表
```

**用戶體驗**：
```bash
# 場景 1: 新用戶快速測試
git clone https://github.com/user/handover-rl.git
./setup_env.sh
python train_sb3.py --config configs/config.yaml --seed 42 --episodes 500
# → 使用 7天 LFS 預計算表，快速測試 ✅

# 場景 2: 完整研究訓練
python tools/generate_orbit_precompute.py --duration-days 30
python train_sb3.py --config configs/config.yaml --seed 42 --episodes 2500
# → 使用 30天本地表，完整訓練 ✅
```

---

## 結論

**14天新鮮度**指的是：
- ✅ 衛星池 TLE epoch 的年齡（30KB 文件）
- ❌ 不是預計算表的覆蓋範圍

**預計算表可以覆蓋 30 天**，但需要：
1. 生成時使用新鮮 TLE（0-7天）
2. 總誤差時長（TLE age + 預測時長）控制在 30-45 天內
3. 定期重新生成（推薦每 2-4 週）

**實際可用的數據策略**：
- **30KB 衛星池**（Git 追蹤）+ **600MB 7天預計算**（Git LFS）+ **2.6GB 30天預計算**（本地生成）
- 用戶可根據需求選擇使用哪一層數據
