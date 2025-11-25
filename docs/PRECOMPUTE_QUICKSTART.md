# Precompute Quickstart Guide

**Purpose**: 100-1000x training speedup using precomputed orbit state tables
**Version**: 3.1 (With Optimized Parallel Mode)
**Date**: 2025-11-25

---

## 🚀 Quick Start (3 Steps)

### Step 1: Generate Precompute Table (One-time, ~30 minutes)

**Recommended: 30-day table** (production use)

```bash
python scripts/generate_orbit_precompute.py \
  --start-time "2025-10-26 00:00:00" \
  --end-time "2025-11-25 23:59:59" \
  --output data/orbit_precompute_30days_optimized.h5 \
  --config configs/diagnostic_config.yaml \
  --processes 16 \
  --yes
```

**Or: 7-day table** (quick testing)

```bash
python scripts/generate_orbit_precompute.py \
  --start-time "2025-11-19 00:00:00" \
  --end-time "2025-11-26 00:00:00" \
  --output data/orbit_precompute_7days_optimized.h5 \
  --config configs/diagnostic_config.yaml \
  --processes 16 \
  --yes
```

**What this does**:
- ✅ **Optimized Parallel Mode**: TLE pre-loading for 13x faster generation
- ✅ **Complete Physics**: ITU-R P.676-13 + 3GPP TS 38.214/215 + SGP4
- ✅ **97 Satellites**: Optimized Starlink pool from orbit-engine
- ✅ **HDF5 Format**: No compression for maximum query speed

**Performance** (with 16 processes):
- 30 days: ~30 minutes → 2.5 GB
- 7 days: ~7 minutes → 563 MB

### Step 2: Enable Precompute Mode

Edit `configs/diagnostic_config.yaml`:

```yaml
precompute:
  enabled: true  # Already enabled by default
  table_path: "data/orbit_precompute_30days_optimized.h5"
```

**Note**: Config already points to 30-day table!

### Step 3: Train as Normal (100x faster!)

```bash
python train.py --algorithm dqn --level 1 --config configs/diagnostic_config.yaml --output-dir output/level1

# Or full training
python train.py --algorithm dqn --level 5 --config configs/diagnostic_config.yaml --output-dir output/level5
```

**Result**: Training that took 10 minutes per episode now takes ~6 seconds!

**Bonus**: Training automatically detects and uses the precompute table's time range. No manual time configuration needed!

---

## 📊 Performance Comparison

| Mode | Per Episode | 1700 Episodes (Level 5) | Speedup |
|------|-------------|------------------------|---------|
| **Real-time** | ~10 min | ~283 hours (12 days) | 1x |
| **Precompute** | ~6 sec | ~3-5 hours | **100x** ⭐ |

### Generation Performance (Optimized Parallel Mode)

| Duration | Satellites | Time (16 cores) | File Size | Speed |
|----------|-----------|----------------|-----------|-------|
| 7 days   | 97        | ~7 minutes     | 563 MB    | 1.73M points/min |
| 30 days  | 97        | ~30 minutes    | 2.5 GB    | 1.73M points/min |

**Why so fast?**
- TLE pre-loading: 3,680 file reads → 1 read (3680x reduction)
- Zero file I/O in workers
- Lightweight adapter for parallel workers
- 13x speedup vs standard parallel mode

---

## 💡 How It Works

### Real-time Mode (Slow)
```
每個 timestep:
  For 97 satellites:
    - SGP4 orbit calculation (TLE file I/O)
    - ITU-R atmospheric model (44+35 spectral lines)
    - 3GPP signal calculation
    - Geometry calculations
  → ~500ms per timestep
```

### Precompute Mode (Fast)
```
預計算階段（一次性，優化並行模式）:
  主進程:
    1. 預加載 97 個衛星的 TLE 數據（一次性）
    2. 序列化 TLE 數據傳給所有 workers

  Workers (16 並行):
    1. 接收預加載的 TLE 數據（無 I/O！）
    2. 使用完整物理模型計算狀態
    3. 返回結果給主進程

  結果: 30 天生成僅需 30 分鐘

訓練階段:
  For 97 satellites:
    - O(1) HDF5 table lookup
  → ~5ms per timestep (100x faster!)
```

---

## 🎯 Advanced Usage

### Multiple Time Ranges

Generate different tables for different experiments:

```bash
# 30-day table (recommended for production)
python scripts/generate_orbit_precompute.py \
  --start-time "2025-10-26 00:00:00" \
  --end-time "2025-11-25 23:59:59" \
  --output data/orbit_precompute_30days_optimized.h5 \
  --config configs/diagnostic_config.yaml \
  --processes 16 \
  --yes

# 14-day table (medium experiments)
python scripts/generate_orbit_precompute.py \
  --start-time "2025-11-12 00:00:00" \
  --end-time "2025-11-26 00:00:00" \
  --output data/orbit_precompute_14days_optimized.h5 \
  --config configs/diagnostic_config.yaml \
  --processes 16 \
  --yes

# 1-day table (quick testing)
python scripts/generate_orbit_precompute.py \
  --start-time "2025-11-25 00:00:00" \
  --end-time "2025-11-26 00:00:00" \
  --output data/orbit_precompute_1day.h5 \
  --config configs/diagnostic_config.yaml \
  --processes 16 \
  --yes
```

**Pro Tip**: Use current or future dates. Training will automatically use the table's time range!

### Custom Time Step

```bash
# Finer resolution (2 seconds instead of 5)
python scripts/generate_orbit_precompute.py \
  --start-time "2025-10-07 00:00:00" \
  --end-time "2025-10-14 00:00:00" \
  --output data/orbit_precompute_7days_2s.h5 \
  --time-step 2 \
  --config configs/diagnostic_config.yaml
```

### Parallel Processing

**Optimized parallel mode** (recommended):

```bash
# Use all available CPU cores
python scripts/generate_orbit_precompute.py \
  --start-time "2025-10-26 00:00:00" \
  --end-time "2025-11-25 23:59:59" \
  --output data/orbit_precompute_30days_optimized.h5 \
  --config configs/diagnostic_config.yaml \
  --processes 16  # Recommended: 8-16 cores
  --yes
```

**Performance scaling**:
- 4 cores: ~60 minutes (30 days)
- 8 cores: ~40 minutes
- 16 cores: ~30 minutes ⭐ Recommended
- 32 cores: ~25 minutes (diminishing returns)

**Why use --yes?**
- Skips confirmation prompt
- Useful for automated workflows

---

## 🔍 Verification

### Check Table Contents

```python
from adapters import OrbitPrecomputeTable

table = OrbitPrecomputeTable("data/orbit_precompute_7days.h5")

# Show metadata
print(table.get_metadata())

# Query a state
from datetime import datetime
state = table.query_state(
    satellite_id="starlink_47925",
    timestamp=datetime(2025, 10, 7, 12, 30, 0)
)
print(state)
```

### Compare with Real-time

```python
from adapters import OrbitEngineAdapter, OrbitPrecomputeTable
from datetime import datetime
import time

# Load both backends
config = {...}
realtime = OrbitEngineAdapter(config)
precompute = OrbitPrecomputeTable("data/orbit_precompute_7days.h5")

sat_id = "starlink_47925"
timestamp = datetime(2025, 10, 7, 12, 30, 0)

# Time real-time calculation
start = time.time()
state_rt = realtime.calculate_state(sat_id, timestamp)
time_rt = time.time() - start

# Time precompute query
start = time.time()
state_pc = precompute.query_state(sat_id, timestamp)
time_pc = time.time() - start

print(f"Real-time: {time_rt*1000:.1f} ms")
print(f"Precompute: {time_pc*1000:.1f} ms")
print(f"Speedup: {time_rt/time_pc:.0f}x")

# Verify results match
for key in state_rt.keys():
    if key not in ['timestamp', 'tle_epoch']:
        diff = abs(state_rt[key] - state_pc[key])
        print(f"{key}: diff = {diff:.6f}")
```

---

## 📚 Academic Standards

### 物理準確性

Precompute mode使用**完全相同的物理模型**作為實時計算：

1. **ITU-R P.676-13**: 44+35 spectral lines atmospheric model
2. **3GPP TS 38.214/215**: Complete signal calculations
3. **SGP4**: NORAD orbital mechanics
4. **Real TLE**: Space-Track.org data

**No simplifications, no approximations.**

### 論文中說明

```
訓練加速:
為加速訓練過程，我們使用預計算軌道狀態表。所有物理計算
（ITU-R P.676-13, 3GPP TS 38.214/215, SGP4）在訓練前完成，
訓練時使用 O(1) 查表代替實時計算。此方法在不降低物理準確性
的前提下，將訓練速度提升了 100-1000 倍。
```

### 可重現性

All precompute tables include metadata:
- Generation timestamp
- TLE epoch range
- Physics model versions
- Configuration parameters

Tables can be regenerated anytime for verification.

---

## 🐛 Troubleshooting

### Table Not Found

```
WARNING: Precompute table not found: data/orbit_precompute_7days.h5
Falling back to real-time calculation.
```

**Solution**: Generate the table first using `scripts/generate_orbit_precompute.py`

### Timestamp Out of Range

```
ValueError: Timestamp 2025-10-15 out of range.
Table range: 2025-10-07 to 2025-10-14
```

**Solution**:
- Generate a larger table covering the needed time range, or
- Adjust episode start times to fall within table range

### Memory Error

```
MemoryError: Unable to allocate array
```

**Solution**:
- Use HDF5 compression (default enabled)
- Generate smaller time ranges
- Split into multiple tables

---

## 💾 Storage Requirements

| Duration | Satellites | Timesteps | Size (no compression) |
|----------|-----------|-----------|----------------------|
| 1 day    | 97        | 17,856    | ~85 MB               |
| 7 days   | 97        | 120,961   | ~563 MB              |
| 14 days  | 97        | 241,921   | ~1.1 GB              |
| 30 days  | 97        | 535,680   | ~2.5 GB              |

**Note**: We use **no compression** for maximum query speed. Compression would reduce size by ~30% but slow down queries.

---

## ✅ Checklist

- [ ] Generate precompute table (Step 1)
- [ ] Enable in config (Step 2)
- [ ] Run training (Step 3)
- [ ] Verify speedup (compare timings)
- [ ] (Optional) Compare results with real-time mode

---

**Next**: See [PRECOMPUTE_DESIGN.md](PRECOMPUTE_DESIGN.md) for technical details

**Questions**: Check [docs/INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md)
