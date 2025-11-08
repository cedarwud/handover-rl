# 學術標準訓練加速計畫

## 目標
- ✅ 達到學術發表標準（1M+ steps, multiple seeds, baselines）
- ✅ 加速實驗週期（7x speedup）
- ✅ 不犧牲精準度、可信度、可靠度

---

## 階段 1: 預計算軌道表系統 (3天)

### 1.1 開發預計算工具 (1天)

**檔案**: `scripts/precompute_orbits.py`

```python
"""
預計算衛星軌道表

功能:
- 載入 TLE 數據
- 使用 Skyfield SGP4 計算所有時刻的衛星位置
- 儲存為高效格式（HDF5）
- 包含插值方法（快速查詢）
"""

import h5py
import numpy as np
from skyfield.api import load, wgs84, EarthSatellite
from datetime import datetime, timedelta
from tqdm import tqdm

class OrbitPrecomputer:
    def __init__(self, tle_file: str, output_file: str):
        self.tle_file = tle_file
        self.output_file = output_file

    def precompute(self,
                   duration_days: int = 30,
                   time_step_seconds: int = 5):
        """
        預計算軌道

        Args:
            duration_days: 預計算天數（建議30天）
            time_step_seconds: 時間間隔（5秒匹配環境step）
        """
        # 1. 載入 TLE
        satellites = self._load_tle()

        # 2. 生成時間序列
        time_points = self._generate_time_points(duration_days, time_step_seconds)

        # 3. 計算所有衛星在所有時刻的位置
        positions = self._compute_all_positions(satellites, time_points)

        # 4. 儲存到 HDF5
        self._save_to_hdf5(positions, time_points)

    def _compute_all_positions(self, satellites, time_points):
        """
        計算所有衛星位置

        Returns:
            positions: (num_sats, num_times, 3) array
                      ECEF coordinates in meters
        """
        num_sats = len(satellites)
        num_times = len(time_points)
        positions = np.zeros((num_sats, num_times, 3), dtype=np.float32)

        for i, sat in enumerate(tqdm(satellites, desc="Computing orbits")):
            for j, t in enumerate(time_points):
                geocentric = sat.at(t)
                subpoint = wgs84.geographic_position_of(geocentric)

                # 儲存 ECEF coordinates
                positions[i, j] = geocentric.position.km * 1000  # km to m

        return positions
```

### 1.2 開發環境適配器 (1天)

**檔案**: `src/environments/precomputed_satellite_env.py`

```python
"""
使用預計算軌道的環境

與原始環境完全兼容，但使用查表代替實時計算
"""

import h5py
import numpy as np
from datetime import datetime

class PrecomputedOrbitEnv(SatelliteHandoverEnv):
    def __init__(self, orbit_table_path: str, **kwargs):
        super().__init__(**kwargs)

        # 載入預計算軌道表
        self.orbit_table = h5py.File(orbit_table_path, 'r')
        self.positions = self.orbit_table['positions']  # (num_sats, num_times, 3)
        self.timestamps = self.orbit_table['timestamps'][:]
        self.time_step = self.orbit_table.attrs['time_step_seconds']

        # 建立快速查詢索引
        self._build_time_index()

    def _get_satellite_position(self, sat_id: int, time: datetime):
        """
        從預計算表中查詢衛星位置（而非實時計算）

        使用線性插值確保精度
        """
        # 找到時間索引
        time_idx = self._find_time_index(time)

        # 線性插值（如果不是精確時刻）
        if self._is_exact_time(time, time_idx):
            return self.positions[sat_id, time_idx]
        else:
            return self._interpolate_position(sat_id, time, time_idx)
```

### 1.3 驗證和測試 (1天)

**測試項目**:

1. **精度驗證**:
```python
# 隨機抽樣100個時刻，對比預計算 vs 實時計算
def test_precomputed_accuracy():
    original_env = SatelliteHandoverEnv(...)
    precomputed_env = PrecomputedOrbitEnv(...)

    for _ in range(100):
        random_time = generate_random_time()
        random_sat = random.choice(satellites)

        pos_original = original_env._get_satellite_position(random_sat, random_time)
        pos_precomputed = precomputed_env._get_satellite_position(random_sat, random_time)

        error = np.linalg.norm(pos_original - pos_precomputed)
        assert error < 1.0  # 誤差 < 1 meter
```

2. **速度測試**:
```python
# 確認真的有7x加速
def benchmark_speed():
    import time

    # Original
    start = time.time()
    run_episodes(original_env, n=10)
    time_original = time.time() - start

    # Precomputed
    start = time.time()
    run_episodes(precomputed_env, n=10)
    time_precomputed = time.time() - start

    speedup = time_original / time_precomputed
    print(f"Speedup: {speedup:.1f}x")
    assert speedup > 5.0  # 至少5x
```

3. **環境一致性**:
```python
# 確保環境API完全兼容
def test_env_compatibility():
    # 兩個環境應該產生完全相同的軌跡（相同seed）
    traj_original = run_episode(original_env, seed=42)
    traj_precomputed = run_episode(precomputed_env, seed=42)

    assert np.allclose(traj_original['states'], traj_precomputed['states'])
    assert np.array_equal(traj_original['actions'], traj_precomputed['actions'])
```

**輸出**:
- ✅ 驗證報告（證明精度一致）
- ✅ 速度測試結果
- ✅ 預計算軌道表（~20GB HDF5 file）

---

## 階段 2: 學術標準訓練 (7-10天)

### 2.1 多 Seed 訓練 (7天)

**RL研究標準要求**: 至少 3-5 個不同 random seeds

```bash
# 訓練配置
SEEDS=(42 123 456 789 1024)
STEPS_PER_SEED=1000000  # 1M steps

for seed in "${SEEDS[@]}"; do
    python train.py \
        --env precomputed \
        --orbit-table precomputed_orbits.h5 \
        --seed $seed \
        --total-steps 1000000 \
        --log-dir logs/seed_${seed} \
        --checkpoint-dir checkpoints/seed_${seed}
done
```

**時間估算**:
- 每個seed: 1.4天（使用預計算）
- 5個seeds: 7天
- **不用預計算**: 48天 ❌

### 2.2 結果分析和報告

**論文中報告**:
```latex
\begin{table}[t]
    \centering
    \caption{Training results across 5 random seeds (1M steps each).}
    \begin{tabular}{lcccc}
        \toprule
        Metric & Mean & Std & Min & Max \\
        \midrule
        Final Reward & -650.2 & 85.3 & -753.1 & -542.8 \\
        Best Reward & -275.1 & 25.6 & -288.6 & -245.3 \\
        Convergence Episode & 850 & 120 & 720 & 1020 \\
        \bottomrule
    \end{tabular}
\end{table}
```

---

## 階段 3: Baselines 和 Ablations (15-20天)

### 3.1 Baseline Methods (5天)

**必須的比較**:

1. **Random Policy**: 隨機選擇動作
2. **Greedy Policy**: 總是選最高SINR
3. **Hysteresis**: 傳統 handover 算法（有遲滯）

```python
# 每個baseline跑5個seeds
for method in random greedy hysteresis; do
    for seed in "${SEEDS[@]}"; do
        python train.py \
            --method $method \
            --seed $seed \
            ...
    done
done
```

### 3.2 Ablation Studies (10天)

**測試項目**:

1. **Reward權重**:
   - SINR weight: [0.5, 1.0, 2.0]
   - Handover penalty: [1, 5, 10]

2. **網絡架構**:
   - Hidden dims: [128, 256, 512]
   - Num layers: [2, 3, 4]

3. **訓練超參數**:
   - Learning rate: [1e-4, 3e-4, 1e-3]
   - Batch size: [64, 128, 256]

4. **Numerical Stability Levels**:
   - Level 0 (baseline): 無數值穩定
   - Level 3: 基本穩定
   - Level 5: 完整穩定（我們的方法）

**預估實驗數**:
- Reward: 3×3 = 9 runs
- Network: 3×3 = 9 runs
- Hyperparams: 3×3 = 9 runs
- Stability: 3 runs
- **總計: ~30 runs × 1.4天 = 42天**
- **可並行**: 如果有多GPU，可縮短到2周

---

## 時間和成本估算

### 方案A: 使用預計算（推薦）

| 階段 | 時間 | 產出 |
|------|------|------|
| 開發預計算 | 3天 | 加速基礎設施 |
| 多seed訓練 | 7天 | 5×1M steps |
| Baselines | 5天 | 3種方法對比 |
| Ablations | 15天 | 完整分析 |
| **總計** | **30天** | **發表級研究** |

### 方案B: 不加速（不推薦）

| 階段 | 時間 |
|------|------|
| 多seed訓練 | 48天 |
| Baselines | 30天 |
| Ablations | 100天 |
| **總計** | **178天（~6個月）** |

**節省時間**: 148天 ✅

---

## 學術標準檢查清單

完成後你將有：

### ✅ 訓練標準
- [x] 1M+ steps per experiment
- [x] Multiple random seeds (5個)
- [x] Mean ± Std 報告
- [x] Convergence curves

### ✅ 比較標準
- [x] Baseline methods (3種)
- [x] Ablation studies (完整)
- [x] Statistical significance tests

### ✅ 可重複性
- [x] 完整代碼開源
- [x] 預計算軌道表（可分享）
- [x] 詳細超參數記錄
- [x] 訓練曲線和checkpoint

### ✅ 技術貢獻
- [x] Numerical stability 驗證（Episode 920對比）
- [x] Domain-specific 創新（衛星handover）
- [x] 高效訓練方法（預計算加速）

---

## 論文撰寫要點

### Experiments章節結構

```latex
\section{Experiments}

\subsection{Experimental Setup}
- Environment: 122 LEO satellites (Iridium NEXT)
- Training: 1M steps, 5 random seeds
- Baselines: Random, Greedy, Hysteresis
- Implementation: PyTorch, Gymnasium

\subsection{Numerical Stability Analysis}
- Figure: Episode 920 comparison (baseline explodes, ours stable)
- Table: Loss statistics across training

\subsection{Performance Comparison}
- Table: Our method vs baselines (mean ± std)
- Figure: Learning curves with confidence intervals

\subsection{Ablation Studies}
- Reward weight analysis
- Network architecture impact
- Stability level comparison

\subsection{Domain Analysis}
- Handover frequency trends
- Strategy learning visualization
- Physical interpretation
```

---

## 實施建議

### 立即開始 (Day 1-3)

```bash
# 1. 創建開發分支
git checkout -b feature/precomputed-orbits

# 2. 開始開發預計算系統
mkdir -p scripts/precompute
mkdir -p src/environments/precomputed

# 3. 使用我提供的代碼框架開始實作
```

### 驗證里程碑 (Day 3)

必須確認：
- ✅ 精度測試通過（誤差 < 1m）
- ✅ 速度測試通過（加速 > 5x）
- ✅ 環境兼容性測試通過

### 正式實驗 (Day 4-30)

使用實驗管理工具：
```bash
# 推薦使用 Weights & Biases 或 MLflow
wandb login
python train.py --wandb-project satellite-handover ...
```

---

## 風險和應對

### 風險1: 預計算文件太大
- **預計**: 20-50GB
- **應對**: 使用壓縮（HDF5 gzip）、雲端儲存

### 風險2: 插值精度問題
- **預計**: 線性插值可能不夠精確
- **應對**: 使用三次樣條插值、增加採樣頻率

### 風險3: 開發時間超出
- **預計**: 可能需要4-5天而非3天
- **應對**: 即使5天，總時間仍遠小於不加速

---

## 結論

**強烈建議採用此方案**，因為：

1. ✅ **達到學術標準**: 1M steps, multiple seeds, baselines
2. ✅ **加速顯著**: 節省148天（5個月）
3. ✅ **不犧牲精度**: 可以驗證預計算與原始環境一致
4. ✅ **長期價值**: 未來所有實驗都受益
5. ✅ **投資回報高**: 3天開發換取5個月節省

**時間表**: 今天開始 → 30天後完成發表級完整研究

需要我開始實施嗎？
