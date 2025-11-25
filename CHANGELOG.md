# Changelog

All notable changes to the Handover-RL project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.1.0] - 2025-11-25

### Added
- **Optimized Parallel Mode for Precompute Generation** - 13x faster with TLE pre-loading
  - TLE data now pre-loaded once in main process (not per worker)
  - Reduces file I/O from 3,680 reads to 1 read (3680x reduction)
  - New `OrbitEngineAdapterLightweight` for workers without file I/O
  - New `_precompute_worker_optimized.py` for optimized worker function
  - Fallback chain: optimized → standard parallel → serial mode

- **Automatic Time Range Detection** in Training
  - Training now auto-detects time range from precompute table metadata
  - No manual time configuration needed
  - Supports both precompute and real-time modes seamlessly

- **30-day Precompute Table Generation**
  - Generated complete 30-day table (2025-10-26 to 2025-11-25)
  - 97 satellites, 535,680 timesteps, 2.5 GB
  - Generation time: ~30 minutes with optimized mode

- **Monitoring Script**
  - Added `tools/monitor_30day_generation.sh` for progress tracking

### Changed
- **Precompute Generation Performance**
  - 30 days: 30 minutes (was estimated 232 hours without optimization)
  - 7 days: ~7 minutes (was 90 minutes with failed TLE pre-loading)
  - Speed: 1.73M points/minute (was 130K points/minute)

- **Configuration Updates**
  - `configs/diagnostic_config.yaml` now points to 30-day table by default
  - Updated path: `data/orbit_precompute_30days_optimized.h5`

### Fixed
- **TLE Loader Method Name** in `orbit_precompute_generator.py`
  - Changed from `get_tle()` to `get_tle_for_date()` (correct method)
  - Fixed TLE pre-loading failure that caused 13x slowdown
  - Now successfully pre-loads TLE data for all 97 satellites

- **Timestamp Range Mismatch** in `train.py`
  - Training now correctly reads time range from adapter backend
  - Uses `adapter.get_backend_info()` for unified interface
  - Automatic detection for both precompute and real-time modes

### Performance Improvements
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| File I/O (per generation) | 3,680 reads | 1 read | 3680x reduction |
| 30-day generation time | ~232 hours (estimated) | 30 minutes | 464x faster |
| Generation speed | 130K points/min | 1.73M points/min | 13.3x faster |
| TLE pre-loading | ❌ Failed | ✅ Success | Fixed |

### Technical Details
**Optimization Architecture:**
```
Main Process:
  1. Load TLE data once from 230 files
  2. Serialize TLE data (97 satellites)
  3. Pass to workers via multiprocessing

Workers (16 parallel):
  1. Receive pre-loaded TLE data
  2. Create OrbitEngineAdapterLightweight (no I/O)
  3. Compute states using physics models
  4. Return results to main process
```

**Files Modified:**
- `src/adapters/orbit_precompute_generator.py` - Fixed TLE method name
- `src/adapters/orbit_engine_adapter_lightweight.py` - Lightweight adapter (existing)
- `src/adapters/_precompute_worker_optimized.py` - Optimized worker (existing)
- `train.py` - Auto-detect time range from adapter
- `configs/diagnostic_config.yaml` - Point to 30-day table
- `README.md` - Updated documentation with new performance metrics
- `scripts/generate_orbit_precompute.py` - Updated usage examples

### Verification
- ✅ 30-day precompute table generated successfully
- ✅ Data integrity verified (97 satellites × 535,680 timesteps)
- ✅ Level 1 training test passed (3 episodes in 59 seconds)
- ✅ Time range auto-detection working correctly
- ✅ No timestamp mismatch errors

---

## [3.0.0] - 2024-11-24

### Completed
- ✅ Level 5 Training Complete (1,700 episodes, 35 hours)
- ✅ Level 6 Training Complete (4,174 episodes, 120 hours)
- ✅ **70.6% handover reduction** achieved vs RSRP baseline
- ✅ Precompute system providing 100x training acceleration
- ✅ Paper assets generated (6 PDFs + 1 LaTeX table)

### Major Features
- Complete precompute acceleration system
- ITU-R P.676-13 + 3GPP TS 38.214/215 + SGP4 physics
- Real TLE data from Space-Track.org
- DQN and Double DQN implementations
- Multi-level training strategy (Levels 0-6)

---

## Notes

### Commit History
Recent commits related to v3.1.0:
- `4e6a7bf` - fix: Correct TLE loader method name in precompute generation
- `b5b4443` - fix: Auto-detect time range from precompute table metadata
- `d56ab94` - perf: Optimize parallel precompute generation (100x faster)
- `c547c4d` - deps: Clean up requirements.txt to match actual usage

### Future Work
- Consider generating longer precompute tables (60-90 days) for extended training
- Explore compression options to reduce storage while maintaining speed
- Add automatic precompute table validation on load
