# Data Dependencies - orbit-engine Integration

**目的**: 明確 handover-rl 對 orbit-engine 輸出的依賴關係

---

## 🎯 Critical Dependency: Starlink Satellite Pool (Stage 4)

### What We Need
**101 Starlink satellites from Stage 4 optimization**:
- ✅ **101 Starlink** (550km LEO) - **Used in this project**
- ❌ **24 OneWeb** (1200km LEO) - **Not used** (cross-constellation handover unrealistic)

**Project Scope**: Starlink-only constellation

**Rationale**:
- 根據 2023-2025 文獻調研，**所有 LEO satellite handover 論文都使用單一星座**（Starlink OR OneWeb），從未有跨星座換手的研究
- 跨星座換手在商業上不現實（like AT&T vs Verizon - 不同的運營商、地面站、用戶終端）
- Starlink 提供更多訓練數據：101 satellites vs 24 satellites (4.2倍優勢)

詳見 [CONSTELLATION_CHOICE.md](CONSTELLATION_CHOICE.md) 完整說明

### Where to Get It

**File Path**:
```
/home/sat/satellite/orbit-engine/data/outputs/stage4/link_feasibility_output_YYYYMMDD_HHMMSS.json
```

**JSON Structure**:
```json
{
  "pool_optimization": {
    "optimized_pools": {
      "starlink": [
        {
          "satellite_id": "45540",  // ← Use this as satellite_id
          "name": "45540",
          "constellation": "starlink",
          "time_series": [...],
          "service_window": {...}
        },
        // ... 101 satellites total
      ],
      "oneweb": [
        {
          "satellite_id": "48058",  // ← Use this as satellite_id
          "name": "48058",
          "constellation": "oneweb",
          "time_series": [...],
          "service_window": {...}
        },
        // ... 24 satellites total
      ]
    }
  }
}
```

### How to Extract (Correct Implementation)

```python
import json
from pathlib import Path

def load_stage4_starlink_satellites() -> list[str]:
    """
    Load 101 Starlink satellites from orbit-engine Stage 4 output

    **Project Scope**: Starlink-only (cross-constellation handover unrealistic)

    Returns:
        List of 101 Starlink satellite IDs (NORAD catalog numbers as strings)

    SOURCE: orbit-engine Stage 4 Pool Optimization
    NO HARDCODING: Reads from actual six-stage processing output
    """
    # Find latest Stage 4 output
    stage4_dir = Path("/home/sat/satellite/orbit-engine/data/outputs/stage4")
    stage4_files = sorted(stage4_dir.glob("link_feasibility_output_*.json"))

    if not stage4_files:
        raise FileNotFoundError(
            f"No Stage 4 output found in {stage4_dir}. "
            "Run orbit-engine stages 1-4 first."
        )

    latest_file = stage4_files[-1]

    with open(latest_file) as f:
        data = json.load(f)

    pools = data['pool_optimization']['optimized_pools']

    # Extract Starlink satellite IDs only
    starlink_pool = pools.get('starlink', [])
    satellite_ids = [sat['satellite_id'] for sat in starlink_pool]

    # Validation
    assert len(satellite_ids) == 101, \
        f"Expected 101 Starlink satellites, got {len(satellite_ids)}"

    return satellite_ids
```

---

## ❌ WRONG APPROACHES (DO NOT USE)

### ❌ Wrong 1: Extract from TLE Files Directly
```python
# ❌ WRONG: Bypasses six-stage optimization
satellite_ids = extract_satellites_from_tle(tle_file, max_satellites=125)
```

**Why Wrong**:
- TLE has 8000+ satellites, but only 125 passed Stage 1-4 optimization
- Skips visibility analysis, service window calculation, pool optimization
- Violates academic requirement: use processed data, not raw data

### ❌ Wrong 2: Extract from TLE by Prefix Only
```python
# ❌ WRONG: Bypasses Stage 4 optimization
satellite_ids = extract_satellites_from_tle(tle_file, prefix="STARLINK", max_satellites=101)
```

**Why Wrong**:
- Bypasses Stage 4 optimization (visibility analysis, service windows)
- Not all Starlink satellites are suitable for handover scenarios
- Should use Stage 4 scientifically-selected pool, not arbitrary TLE selection

### ❌ Wrong 3: Use "Dynamic Pool Selection" Calculations
```python
# ❌ WRONG: Reimplements what Stage 4 already did
pool_size = calculate_from_orbital_mechanics(...)
satellite_ids = extract_satellites_from_tle(tle_file, max_satellites=pool_size)
```

**Why Wrong**:
- Duplicates orbit-engine Stage 4 Pool Optimization logic
- Ignores actual visibility analysis and service windows
- Results won't match Stage 4 scientific selection criteria

---

## ✅ Verification Steps

### Before Running handover-rl Training

```bash
# 1. Check Stage 4 output exists
ls -lh /home/sat/satellite/orbit-engine/data/outputs/stage4/link_feasibility_output_*.json

# 2. Verify Starlink satellite count
python3 -c "
import json
from pathlib import Path

stage4_dir = Path('/home/sat/satellite/orbit-engine/data/outputs/stage4')
latest = sorted(stage4_dir.glob('link_feasibility_output_*.json'))[-1]

with open(latest) as f:
    data = json.load(f)

pools = data['pool_optimization']['optimized_pools']
print(f'Starlink: {len(pools[\"starlink\"])} satellites')
print(f'OneWeb: {len(pools[\"oneweb\"])} satellites (not used in this project)')
print(f'Project uses: {len(pools[\"starlink\"])} Starlink satellites only')
"

# Expected output:
# Starlink: 101 satellites
# OneWeb: 24 satellites (not used in this project)
# Project uses: 101 Starlink satellites only
```

### After Loading Satellites in Training Script

```python
satellite_ids = load_stage4_starlink_satellites()

# Verify
print(f"✅ Loaded {len(satellite_ids)} Starlink satellites")
print(f"   First 3: {satellite_ids[:3]}")  # Should be NORAD IDs like ['45540', '46701', ...]
print(f"   Last 3: {satellite_ids[-3:]}")   # Should be Starlink IDs

# Check for Starlink
assert len(satellite_ids) == 101, "Should have exactly 101 Starlink satellites"
assert '45540' in satellite_ids  # Starlink example
```

---

## 🔄 Relationship with Stage 5 & Stage 6

### Stage 5: Signal Quality Analysis

**What Stage 5 Provides**:
- Signal quality calculation **algorithms** (ITU-R P.676 + 3GPP TS 38.214/38.215)
- RSRP/RSRQ/SINR computation methods
- Pre-computed signal data for fixed time range (JSON output)

**How handover-rl Uses It**:
- ✅ **Uses the algorithms** (ITU-R + 3GPP calculators via OrbitEngineAdapter)
- ❌ **Does NOT read Stage 5 JSON output**

**Why Not Read JSON**:
- Stage 5 JSON = Fixed time range (pre-computed)
- RL training = Arbitrary time points (random exploration)
- **Solution**: Real-time calculation using same algorithms

**Implementation**:
```python
# OrbitEngineAdapter uses Stage 5 algorithms
from orbit_engine.stages.stage5_signal_analysis import GPPSignalCalculator

signal_quality = self.gpp_calc.calculate_complete_signal_quality(
    distance_km=...,
    elevation_deg=...,
    # ... other params
)
# Returns: rsrp_dbm, rsrq_db, rs_sinr_db
```

**Key Point**: Uses **algorithms** ≠ Reads **output files**

---

### Stage 6: Research Optimization & 3GPP Events

**What Stage 6 Provides**:
- 3GPP NTN events (A3/A4/A5/D2) - Traditional rule-based handover
- Pool verification statistics
- ML training data generation (⚠️ **Not implemented** - marked as future work)

**How handover-rl Uses It**:
- ⚠️ **Optional**: 3GPP events for baseline comparison experiments
- ❌ **Not used**: ML training data (not implemented in orbit-engine)
- ✅ **Indirectly**: Pool verification confirms 125-satellite quality

**When to Use Stage 6**:
- For baseline comparison: RL agent vs A3 event-based handover
- For performance evaluation: Compare RL learned policy vs traditional rules

**orbit-engine Documentation Quote**:
```
Stage 6 核心功能:
1. 3GPP 事件檢測 (A3/A4/A5/D2)
2. Pool Verification

未來擴展 (待實作):
3. ML 訓練數據生成  ← Not implemented
4. RL 決策算法        ← Not implemented
```

**Conclusion**: RL training is **independent** from orbit-engine (separate project)

---

## 📋 Dependencies Summary

| Dependency | Source | Usage Method | Verification |
|------------|--------|--------------|--------------|
| **Starlink Satellite Pool** | orbit-engine Stage 4 | Read JSON (satellite IDs) | 101 Starlink satellites |
| **Signal Algorithms** | orbit-engine Stage 5 | Call algorithms (real-time) | ITU-R + 3GPP calculators |
| **3GPP Events (Optional)** | orbit-engine Stage 6 | Read JSON (baseline comparison) | A3/A4/A5/D2 events |
| **TLE Data** | orbit-engine Stage 1 | Via OrbitEngineAdapter | 79 TLE files, 84 days coverage |
| **Ground Station Config** | `config/data_gen_config.yaml` | Via OrbitEngineAdapter | NTPU: 24.9441°N, 121.3714°E |

---

## 🚨 Common Mistakes (Lessons Learned)

### Mistake 1: Confusing "Stage 5" in README.md
- **README.md said**: "from orbit-engine Stage 5" (❌ incorrect)
- **Actually from**: Stage 4 Pool Optimization (✅ correct)
- **Fixed**: 2025-10-19 - README.md corrected to Stage 4

### Mistake 2: Assuming "First 101 from TLE"
- **Wrong assumption**: Take first 101 Starlink satellites alphabetically from TLE
- **Correct**: 101 = result of six-stage optimization (visibility, service windows, etc.)
- **Fix**: Always read from Stage 4 output

### Mistake 3: Including OneWeb Satellites
- **Wrong assumption**: Need multi-constellation diversity
- **Correct**: Project uses Starlink-only (cross-constellation handover unrealistic)
- **Fix**: Extract only Starlink from Stage 4 pool

---

**Date**: 2025-10-19
**Status**: ✅ DOCUMENTED - Use this as reference for all satellite loading

---

## 🔧 Critical Implementation Fixes (2025-10-19)

### Issue 1: TLE Loading for Starlink Satellites
**Problem**: Need TLE data for 101 Starlink satellites from Stage 4
- Starlink TLEs: `../orbit-engine/data/tle_data/starlink/tle`
- OneWeb TLEs: Not needed (project uses Starlink-only)

**Fix**: TLELoader configured for Starlink constellation
```python
# src/adapters/tle_loader.py - Starlink TLE source
TLELoader(tle_sources=[
    ('/path/to/starlink/tle', 'starlink_*.tle')
])
```

**Result**:
- ✅ 79 Starlink TLE files loaded
- ✅ All 101 Stage 4 Starlink satellites found
- ✅ Project scope: Starlink-only (cross-constellation handover unrealistic)

### Issue 2: Training Time Window Mismatch
**Problem**: Training time didn't match Stage 4 service windows
- Stage 4 satellites visible: 2025-10-16 03:00:30 - 03:10:30
- Training start time: 2025-10-07 12:00:00 (9 days before!)
- Result: 0 visible satellites, all episodes reward=-1.00

**Root Cause**: LEO satellites only visible ~10 minutes per pass. Stage 4 analyzed specific service windows, but training used arbitrary time.

**Fix**: Use Stage 4 time range for training
```python
# train_online_rl.py
start_time = datetime(2025, 10, 16, 3, 0, 30)  # Match Stage 4 service window
```

**Result**:
- ✅ Satellites visible (3-10 per episode)
- ✅ Realistic RSRP (-85.6 dBm average)
- ✅ Positive rewards (+0.67 to +0.84)
- ✅ Handover events occurring (2-3 per episode)

**Future Improvement**: For longer training, use multiple service window periods or generate episodes across different satellite passes.

---

## 📊 Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    orbit-engine (6 Stages)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: TLE Loading                                           │
│    ↓ Output: 8,632 satellites with TLE data                     │
│                                                                  │
│  Stage 2: Orbital Propagation (SGP4)                            │
│    ↓ Output: TEME coordinates                                   │
│                                                                  │
│  Stage 3: Coordinate Transformation                             │
│    ↓ Output: WGS84 coordinates + visibility                     │
│                                                                  │
│  Stage 4: Pool Optimization ⭐ handover-rl READS THIS           │
│    ↓ Output: 125 satellite IDs (101 Starlink + 24 OneWeb)       │
│    ↓ JSON: pool_optimization['optimized_pools']                 │
│                                                                  │
│  Stage 5: Signal Quality Analysis ⭐ handover-rl USES ALGORITHMS│
│    ↓ Algorithms: ITU-R P.676 + 3GPP TS 38.214/38.215           │
│    ↓ JSON: Pre-computed RSRP/RSRQ/SINR (fixed time range)      │
│                                                                  │
│  Stage 6: 3GPP Events ⭐ handover-rl OPTIONAL BASELINE          │
│    ↓ Output: A3/A4/A5/D2 events for comparison                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              handover-rl (Independent RL Training)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Load 125 Satellite IDs                                      │
│     ← FROM: Stage 4 JSON (pool_optimization)                    │
│                                                                  │
│  2. OrbitEngineAdapter                                          │
│     ├─ Uses Stage 5 algorithms (real-time calculation)          │
│     ├─ Calls: ITU-R P.676, 3GPP TS 38.214/38.215               │
│     └─ Computes: RSRP/RSRQ/SINR for any time point             │
│                                                                  │
│  3. Environment State (12 dimensions)                           │
│     ├─ Signal Quality (5): RSRP, RSRQ, SINR, offsets           │
│     └─ Physical Params (7): distance, elevation, doppler, ...   │
│                                                                  │
│  4. DQN Agent                                                   │
│     ├─ Learns: Handover policy from experience                  │
│     └─ Does NOT use: Stage 6 A3 events (learns own strategy)    │
│                                                                  │
│  5. (Optional) Baseline Comparison                              │
│     └─ Compare with: Stage 6 A3 event-based handover            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Relationships**:
- **Stage 4 → handover-rl**: Provides satellite pool (read JSON)
- **Stage 5 → handover-rl**: Provides algorithms (call functions, not read JSON)
- **Stage 6 → handover-rl**: Optional baseline (A3 events for comparison)

**Why This Architecture**:
- orbit-engine = Offline data processing (fixed time range)
- handover-rl = Online RL training (arbitrary time points)
- Need real-time calculation → Use algorithms, not pre-computed data

---

**Last Updated**: 2025-10-19
**Status**: ✅ Complete data dependency documentation
