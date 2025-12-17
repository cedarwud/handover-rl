# Handover-RL Architecture

本文檔描述 handover-rl 項目的系統架構設計。

---

## 目錄

- [系統概述](#系統概述)
- [依賴關係](#依賴關係)
- [數據架構](#數據架構)
- [模塊設計](#模塊設計)
- [訓練流程](#訓練流程)

---

## 系統概述

Handover-RL 採用**明確依賴架構**：

```
orbit-engine（依賴項目）
    ↓ 提供衛星池數據（Stage 4）
handover-rl（主項目）
    ↓ 加載衛星池
    ↓ 可選生成預計算表（加速）
    ↓ DQN 訓練
```

### 設計原則

1. **職責分離**：orbit-engine 處理物理計算，handover-rl 處理 RL 訓練
2. **明確依賴**：不假裝獨立，清楚說明需要 orbit-engine
3. **數據透明**：明確顯示數據來源和新鮮度
4. **可選加速**：預計算表可選，無預計算也能訓練

---

## 依賴關係

### orbit-engine（必需依賴）

**位置**：必須是 handover-rl 的 sibling directory

```
/home/sat/satellite/
├── orbit-engine/      ← 必需依賴
│   └── data/outputs/stage4/
│       └── link_feasibility_output_*.json  (~29MB)
│
└── handover-rl/       ← 主項目
    └── src/utils/satellite_utils.py  (讀取 Stage 4)
```

**提供內容**：
- **Stage 4 輸出**：`link_feasibility_output_*.json`
- **衛星池**：101 個 Starlink 衛星（科學篩選）
- **元數據**：TLE epoch、衛星軌道參數

**更新頻率**：建議每 2 週（保持 TLE 新鮮度 ≤14 天）

### Python 依賴

- Python 3.10+
- PyTorch 2.0+
- Stable-Baselines3 2.0+
- Gymnasium 0.29+
- H5py（預計算表）

詳見 `requirements.txt`

---

## 數據架構

### 兩種不同的數據文件

這是理解系統的關鍵！

#### 數據 1：衛星池（Satellite Pool）

```
來源：orbit-engine Stage 4
文件：link_feasibility_output_*.json
大小：~29 MB
位置：/home/sat/satellite/orbit-engine/data/outputs/stage4/
```

**內容**：
- 101 個 Starlink 衛星 ID
- 基本軌道參數（高度、傾角）
- TLE epoch（測量日期）

**用途**：
- 告訴系統「有哪些衛星可用」
- 提供基本軌道信息

**新鮮度要求**：≤14 天（推薦）
- 原因：決定衛星池組成
- 衛星可能退役、重新定位

**Git 追蹤**：❌ NO（由 orbit-engine 管理）

---

#### 數據 2：預計算軌道表（Orbit Precompute Table）

```
來源：本地生成（從 orbit-engine 數據）
文件：orbit_precompute_30days.h5
大小：~2.6 GB
位置：handover-rl/data/orbit_precompute/
```

**內容**：
- 30 天 × 518,400 時間步（5 秒間隔）
- 102 衛星 × 每個時間步的完整狀態
  - 位置 (x, y, z)
  - 速度 (vx, vy, vz)
  - 仰角、方位角
  - RSRP（信號強度）
  - 可見性狀態

**用途**：
- 訓練時快速查詢（O(1)）
- 替代實時計算（100-1000x 加速）
- 包含完整物理計算結果

**新鮮度**：總誤差時長 = TLE age + 預測時長
- 例如：TLE 14天前 + 預測未來30天 = 44天總誤差（可接受）

**Git 追蹤**：❌ NO（太大，本地生成）

---

### 數據新鮮度關係

```
TLE Epoch (orbit-engine):     2025-11-26
今天:                         2025-12-17
TLE Age:                      21 天 ← 衛星池新鮮度

預計算開始時間:                2025-12-17
預計算結束時間:                2026-01-16
預測時長:                     30 天 ← 覆蓋範圍

總誤差時長（預測末期）:        21 + 30 = 51 天
```

**建議策略**：
1. 使用最新 TLE 生成預計算表（TLE age < 7 天）
2. 預計算覆蓋 14-30 天
3. 總誤差時長控制在 30-45 天內

---

## 模塊設計

### 目錄結構

```
src/
├── adapters/           # 軌道計算後端
│   ├── adapter_wrapper.py
│   │   └── 自動選擇預計算/實時模式
│   │
│   ├── orbit_engine_adapter.py
│   │   └── 實時物理計算（慢，orbit-engine API）
│   │
│   ├── orbit_precompute_table.py
│   │   └── 快速 O(1) 查詢（HDF5）
│   │
│   └── orbit_precompute_generator.py
│       └── 生成預計算表（一次性）
│
├── environments/       # RL 環境
│   └── satellite_handover_env.py
│       └── V9 環境（RVT-based 獎勵）
│
└── utils/              # 工具
    ├── satellite_utils.py
    │   └── 從 orbit-engine Stage 4 載入衛星池
    │
    └── data_freshness.py (待實現)
        └── 數據新鮮度檢查
```

### 核心模塊

#### 1. satellite_utils.py

**職責**：從 orbit-engine 載入衛星池

```python
def load_stage4_optimized_satellites(
    constellation_filter='starlink'
) -> List[str]:
    """
    從 orbit-engine Stage 4 載入衛星池
    
    路徑：/home/sat/satellite/orbit-engine/data/outputs/stage4/
    
    Returns:
        satellite_ids: List[str] - 101 個衛星 ID
    
    Raises:
        FileNotFoundError: orbit-engine Stage 4 不存在
        DataFreshnessError: 數據過舊（>14 天）
    """
```

**特性**：
- 自動找最新的 Stage 4 文件
- 檢查數據新鮮度
- 驗證衛星數量（101 ± 15）

#### 2. adapter_wrapper.py

**職責**：透明切換預計算/實時模式

```python
class AdapterWrapper:
    def __init__(self, config):
        if precompute_enabled and precompute_exists:
            self.adapter = PrecomputeTable()  # 快速模式
        else:
            self.adapter = OrbitEngineAdapter()  # 實時模式
    
    def get_satellite_state(self, sat_id, time):
        return self.adapter.get_state(sat_id, time)
```

**優點**：
- 訓練代碼無需關心使用哪個後端
- 自動選擇最佳模式
- 無性能開銷

#### 3. satellite_handover_env.py

**職責**：Gymnasium 兼容的 RL 環境

```python
class SatelliteHandoverEnv(gymnasium.Env):
    observation_space: (15, 14)  # 15 衛星 × 14 特徵
    action_space: Discrete(16)   # stay + switch to 15
    
    def step(self, action):
        # 1. 執行動作（切換或保持）
        # 2. 計算 RVT-based 獎勵
        # 3. 檢查駐留時間約束（60s）
        # 4. 返回 (obs, reward, done, info)
```

**V9 特性**：
- RVT-based 獎勵（IEEE TAES 2024）
- 駐留時間約束（防止乒乓切換）
- 動作屏蔽（無效動作）
- 負載感知決策

---

## 訓練流程

### 完整流程圖

```
1. 啟動訓練
   ↓
2. 載入配置（config.yaml）
   ↓
3. 載入衛星池（satellite_utils.py）
   ├─ 讀取 orbit-engine Stage 4
   ├─ 檢查數據新鮮度（≤14 天）
   └─ 驗證衛星數量（101 個）
   ↓
4. 初始化環境（satellite_handover_env.py）
   ├─ 初始化 adapter_wrapper
   │   ├─ 檢查預計算表是否存在
   │   ├─ 存在 → PrecomputeTable（快速）
   │   └─ 不存在 → OrbitEngineAdapter（慢）
   └─ 設置初始狀態
   ↓
5. 初始化 DQN Agent（Stable-Baselines3）
   ├─ 創建神經網絡（MLP [128, 128]）
   ├─ 創建經驗回放緩衝區（10,000）
   └─ 設置超參數
   ↓
6. 訓練循環（2500 episodes）
   ├─ 對每個 episode：
   │   ├─ reset() 環境
   │   ├─ 對每個 timestep（600 秒 / 5 秒 = 120 步）：
   │   │   ├─ 選擇動作（ε-greedy）
   │   │   ├─ step(action)
   │   │   ├─ 存儲經驗到回放緩衝區
   │   │   └─ 訓練神經網絡（批量 64）
   │   └─ 記錄 episode 指標
   └─ 定期保存檢查點（每 500 episodes）
   ↓
7. 保存最終模型
   └─ output/academic_seed42/models/dqn_final.zip
```

### 關鍵性能差異

| 階段 | 有預計算表 | 無預計算表 | 差異 |
|------|-----------|-----------|------|
| **Step 執行** | ~0.001 秒 | ~0.1 秒 | 100x |
| **Episode** | ~0.12 秒 | ~12 秒 | 100x |
| **2500 Episodes** | ~25 分鐘 | ~20 小時 | 48x |

**瓶頸**：`adapter.get_satellite_state()` 調用
- 每個 timestep 調用 15 次（15 個候選衛星）
- 每個 episode 120 timesteps
- 總計：15 × 120 × 2500 = 4,500,000 次調用

---

## 快照機制（可重現性）

### 用途

為論文提交創建數據快照，記錄使用的確切衛星池版本。

### 快照內容

```json
{
  "metadata": {
    "version": "1.0.0",
    "purpose": "IEEE TAES 2025 paper",
    "tle_epoch": "2025-11-26",
    "snapshot_date": "2025-12-17",
    "data_age_days": 21
  },
  "satellite_pools": {
    "starlink": [
      {"satellite_id": "12345", "name": "STARLINK-1234", ...},
      // ... 101 satellites
    ]
  }
}
```

### Git 追蹤策略

| 文件類型 | 大小 | Git 追蹤 | 用途 |
|---------|------|---------|------|
| **快照** | ~30 KB | ✅ YES | 論文可重現性 |
| **預計算表** | ~2.6 GB | ❌ NO | 本地生成 |
| **Stage 4 JSON** | ~29 MB | ❌ NO | orbit-engine 管理 |

### 創建快照

```bash
python tools/data/create_satellite_pool_snapshot.py \
  --version 1.0.0 \
  --description "IEEE TAES 2025 paper"

git add data/satellite_pool/snapshot_v1.0.*
git commit -m "Add snapshot v1.0.0 for paper reproducibility"
```

---

## 設計權衡

### 為什麼依賴 orbit-engine？

**優點**：
- ✅ 單一數據源（避免不一致）
- ✅ 物理計算由專門項目處理
- ✅ 自動獲得 orbit-engine 更新
- ✅ 職責分離清晰

**缺點**：
- ❌ 無法完全獨立運行
- ❌ 設置稍複雜（需要兩個項目）

**結論**：優點遠大於缺點（學術嚴謹性）

---

### 為什麼預計算表可選？

**設計考慮**：
1. **靈活性**：用戶可選擇速度 vs 便利性
2. **磁盤空間**：不是所有用戶都有 5GB 可用空間
3. **快速測試**：小規模測試不需要預計算表
4. **教學演示**：實時模式可展示完整物理計算

**實踐建議**：
- 測試/演示：可選預計算表
- 完整訓練：強烈推薦預計算表
- 論文實驗：必須使用預計算表（時間效率）

---

## 擴展指南

### 添加新星座（OneWeb）

```python
# 1. 修改 config.yaml
environment:
  constellation_filter: 'oneweb'  # 改為 'oneweb'

# 2. 重新載入衛星池（自動）
satellite_ids = load_stage4_optimized_satellites(
    constellation_filter='oneweb'  # 24 OneWeb 衛星
)

# 3. 重新生成預計算表
python tools/orbit/generate_orbit_precompute.py \
  --constellation oneweb \
  --duration-days 30
```

### 添加新獎勵函數

```python
# src/environments/satellite_handover_env.py

def _compute_reward(self, action):
    if self.config['reward_version'] == 'v9':
        return self._rvt_based_reward(action)  # 當前
    elif self.config['reward_version'] == 'v10':
        return self._your_new_reward(action)   # 新函數
```

### 使用不同 RL 算法

```python
# train_sb3.py

# 當前：DQN
from stable_baselines3 import DQN
model = DQN("MlpPolicy", env, ...)

# 改為 PPO
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, ...)
```

---

## 相關文檔

- [README.md](README.md) - 主要文檔（安裝和使用）
- [TLE_FRESHNESS_ANALYSIS.md](TLE_FRESHNESS_ANALYSIS.md) - 數據新鮮度詳細分析
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - 故障排除指南
- [PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md) - 性能優化指南

---

**最後更新**：2025-12-17
