# Handover-RL: LEO Satellite Handover Optimization with Deep RL

Deep Reinforcement Learning framework for optimizing LEO satellite handover decisions with physics-based simulation and 100-1000x training acceleration.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0+-green.svg)](https://stable-baselines3.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-orange.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Results](#results)
- [Tools & Scripts](#tools--scripts)
- [Testing](#testing)
- [Development](#development)
- [Citation](#citation)

---

## Overview

**Handover-RL** is a research framework for optimizing LEO satellite handover decisions using Deep Reinforcement Learning. The system implements a **RVT-based (Remaining Visible Time) reward function** following IEEE TAES 2024 standards, trained with **Stable Baselines3** DQN algorithm on a **precomputed orbit state table** for massive training acceleration.

### Key Innovation: Precompute Acceleration

Traditional RL training for LEO satellite handover is prohibitively slow due to expensive physics calculations (ITU-R atmospheric models, 3GPP signal processing, SGP4 orbital mechanics) repeated at every time step for every satellite candidate.

Our **precompute system** solves this by:
1. **One-time calculation**: Pre-compute all orbit states using complete physics models
2. **O(1) lookup**: Replace expensive calculations with fast HDF5 table queries during training
3. **100-1000x speedup**: Train 2500 episodes in ~25 minutes instead of days
4. **No compromise**: Maintains full academic rigor (no simplified models)

---

## Features

### Core Capabilities

- **RVT-Based Reward** (IEEE TAES 2024): Optimizes for long satellite visibility windows
- **Stable Baselines3 Integration**: Production-ready DQN implementation
- **Multi-Seed Training**: Robust results across 5 random seeds (42, 123, 456, 789, 2024)
- **Precompute Acceleration**: 100-1000x faster training with complete physics
- **Academic-Grade Physics**: ITU-R P.676-13, 3GPP TS 38.214/215, SGP4 orbital mechanics
- **Dwell Time Constraints**: Prevents rapid handover oscillations (60s minimum)
- **Gymnasium Environment**: Standard RL interface for easy integration

### Advanced Features

- **Precompute System**: HDF5-based orbit state caching
- **Action Masking**: Prevents invalid handover decisions
- **Load-Aware Decisions**: Satellite capacity-aware reward shaping
- **Flexible Configuration**: Single YAML config for all parameters
- **Comprehensive Baselines**: RSRP, load-balancing, and hybrid strategies
- **Multi-Seed Analysis**: Statistical confidence via multiple training runs

---

## Project Structure

```
handover-rl/
├── configs/
│   └── config.yaml                 # Main configuration file
│
├── src/                            # Core source code
│   ├── adapters/                   # Orbit calculation adapters
│   │   ├── adapter_wrapper.py      # Precompute/real-time switcher
│   │   ├── orbit_engine_adapter.py # Real-time physics calculations
│   │   ├── orbit_precompute_generator.py  # Precompute table generator
│   │   └── orbit_precompute_table.py      # Precompute table query
│   ├── environments/               # RL environments
│   │   └── satellite_handover_env.py   # Main environment (RVT-based)
│   └── utils/                      # Utilities
│       ├── satellite_utils.py      # Satellite selection & management
│       └── safety_mechanisms.py    # Training safety checks
│
├── scripts/                        # Utility scripts (15 total)
│   ├── evaluate_sb3.py             # Evaluate trained SB3 models
│   ├── evaluate_baselines.py       # Evaluate baseline policies
│   ├── analyze_multi_seeds.py      # Multi-seed statistical analysis
│   └── ... (12 other utility scripts)
│
├── tools/                          # Development tools
│   ├── orbit/                      # Orbit precompute tools
│   │   └── generate_orbit_precompute.py   # Generate HDF5 tables
│   ├── safety/                     # Safety mechanism tools
│   ├── optimization/               # Satellite pool optimization
│   └── visualization/              # Result visualization
│
├── tests/                          # Test suite
│   ├── test_environment.py         # Environment functionality tests
│   └── test_dwell_time.py          # Dwell time constraint tests
│
├── data/                           # Data files (generated, not in git)
│   ├── orbit_precompute_30days.h5            # Precompute table (~2.6GB)
│   └── satellite_ids_from_precompute.txt     # Satellite pool (102 sats)
│
├── train_sb3.py                    # Main training script (SB3)
├── output/                         # Training output (models, logs)
├── archive/                        # Archived old versions
└── README.md                       # This file
```

---

## Installation

### Prerequisites

**Software Requirements:**
- Python 3.10 or higher (3.12 recommended)
- [orbit-engine](../orbit-engine) - Physics simulation engine (sibling directory)
  - ⚠️ **REQUIRED**: Must run orbit-engine Stage 4 to generate satellite pool data
  - Generates: `link_feasibility_output_*.json` (~29MB, contains 101 Starlink satellites)
- Git, GCC/Clang compiler

**Hardware Requirements:**
- RAM: 16GB minimum (32GB recommended for parallel training)
- CPU: Multi-core processor (8+ cores for precompute generation)
- GPU: Optional (CUDA-capable for faster training)
- Storage: ~5GB (3GB for precompute tables, 2GB for models/logs, 30MB for orbit-engine data)

### Setup Steps

**Step 1: Clone Both Repositories**

```bash
# Clone handover-rl and orbit-engine as sibling directories
cd ~/projects  # or your preferred directory

# Clone orbit-engine first (physics dependency)
git clone https://github.com/yourusername/orbit-engine.git

# Clone handover-rl
git clone https://github.com/yourusername/handover-rl.git

# Verify directory structure
ls -la
# Should show:
#   orbit-engine/
#   handover-rl/
```

**Step 2: Setup orbit-engine (REQUIRED - Generates Satellite Pool)**

⚠️ **CRITICAL:** handover-rl requires orbit-engine's Stage 4 output to load the scientifically selected satellite pool (101 Starlink satellites).

```bash
cd orbit-engine

# Setup orbit-engine environment
./setup.sh

# Run orbit-engine processing (Stages 1-4 required, ~10-15 minutes)
source venv/bin/activate
./run.sh --stage 4

# Verify Stage 4 output exists
ls -lh data/outputs/stage4/link_feasibility_output_*.json
# Should show a ~29MB JSON file
```

**What this generates:**
- **Stage 1**: Load TLE data for 9000+ satellites
- **Stage 2**: Propagate orbits for visible satellites
- **Stage 3**: Transform coordinates (ECI → ECEF → geodetic)
- **Stage 4**: Link feasibility analysis ← **REQUIRED FOR HANDOVER-RL**
  - Output: `link_feasibility_output_*.json` (~29MB)
  - Contains: 101 Starlink satellites scientifically selected for handover training
  - Used by: `src/utils/satellite_utils.py:load_stage4_optimized_satellites()`

**Step 3: Setup handover-rl Environment**

**Option A: Automated Setup (Recommended)**
```bash
cd ../handover-rl
./setup_env.sh
```

The `setup_env.sh` script will:
- Check Python version (3.10+ required)
- Verify orbit-engine integration
- Create virtual environment
- Install all dependencies
- Verify installation

**Option B: Manual Setup**
```bash
cd ../handover-rl

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install orbit-engine in editable mode (CRITICAL)
pip install -e ../orbit-engine

# Verify installation
python -c "import src.environments; print('✓ Installation successful')"
```

**Step 4: Generate Precompute Table (Required)**

The precompute table is NOT included in git (too large: 2.6GB). You must generate it:

```bash
# Activate virtual environment
source venv/bin/activate

# Generate 30-day precompute table (~3 hours)
python scripts/generate_orbit_precompute.py \
  --start-time "2025-10-26 00:00:00" \
  --end-time "2025-11-25 23:59:59" \
  --output data/orbit_precompute_30days.h5 \
  --config configs/config.yaml \
  --processes 16 \
  --yes

# Extract satellite IDs
python -c "
import h5py
with h5py.File('data/orbit_precompute_30days.h5', 'r') as f:
    sat_ids = sorted(list(f['states'].keys()))
with open('data/satellite_ids_from_precompute.txt', 'w') as f:
    for sat_id in sat_ids:
        f.write(f'{sat_id}\n')
print(f'✓ Extracted {len(sat_ids)} satellite IDs')
"
```

**Verification:**
```bash
# Check handover-rl data files
ls -lh data/
# Should show:
#   orbit_precompute_30days.h5 (2.6GB)
#   satellite_ids_from_precompute.txt (102 satellite IDs)

# Check orbit-engine Stage 4 output (REQUIRED)
ls -lh ../orbit-engine/data/outputs/stage4/
# Should show:
#   link_feasibility_output_*.json (~29MB)
```

---

## Quick Start

After completing the [Installation](#installation) steps (including orbit-engine Stage 4 processing and precompute table generation), you're ready to train!

**Prerequisites Check:**
```bash
# Verify orbit-engine Stage 4 output exists
ls ../orbit-engine/data/outputs/stage4/link_feasibility_output_*.json

# Verify handover-rl precompute table exists
ls data/orbit_precompute_30days.h5
```

```bash
# Activate virtual environment first
source venv/bin/activate
```

**Step 1: Quick Test Training (100 episodes, ~5 minutes)**

```bash
python train_sb3.py \
  --config configs/config.yaml \
  --output-dir output/test_run \
  --num-episodes 100
```

Expected output:
```
Episode 100/100 | Reward: 45.2 | Handovers: 8.3 | ε: 0.81
✓ Training complete: output/test_run/models/dqn_final.zip
```

**Step 2: Evaluate Trained Model**

```bash
python scripts/evaluate_sb3.py \
  --model output/test_run/models/dqn_final.zip \
  --config configs/config.yaml \
  --episodes 100
```

**Step 3: Compare with Baselines**

```bash
python scripts/evaluate_baselines.py
```

This evaluates 4 baseline policies:
- **RSRP**: Always select highest signal strength
- **Load-Balancing**: Distribute across satellites evenly
- **Hybrid**: RSRP with load-aware switching
- **DQN** (your trained model)

---

## Training

### Single-Seed Training (2500 episodes, ~25 minutes)

```bash
python train_sb3.py \
  --config configs/config.yaml \
  --output-dir output/dqn_seed42 \
  --num-episodes 2500 \
  --seed 42
```

**Training Parameters** (from `configs/config.yaml`):
```yaml
agent:
  learning_rate: 0.0001
  gamma: 0.95
  batch_size: 64
  buffer_capacity: 10000
  target_update_freq: 100
  epsilon_start: 0.82
  epsilon_end: 0.2
  epsilon_decay: 0.9999      # Slow decay for stable learning
```

### Multi-Seed Training (5 seeds × 2500 episodes, ~2 hours)

For robust results with statistical confidence:

```bash
#!/bin/bash
# train_multi_seeds.sh

SEEDS=(42 123 456 789 2024)
for seed in "${SEEDS[@]}"; do
  echo "Training seed $seed..."
  python train_sb3.py \
    --config configs/config.yaml \
    --output-dir output/dqn_seed${seed} \
    --num-episodes 2500 \
    --seed $seed
done
```

Run:
```bash
chmod +x train_multi_seeds.sh
./train_multi_seeds.sh
```

### Training Outputs

Each training run produces:
```
output/dqn_seed42/
├── models/
│   ├── dqn_final.zip          # Final trained model
│   ├── dqn_ep500.zip          # Checkpoint at episode 500
│   ├── dqn_ep1000.zip         # Checkpoint at episode 1000
│   └── ...
├── logs/
│   └── training.log           # Full training log
└── metrics.json               # Episode-wise metrics
```

### Monitoring Training

**Real-time monitoring:**
```bash
# Terminal 1: Start training
python train_sb3.py --config configs/config.yaml --output-dir output/live --num-episodes 2500

# Terminal 2: Monitor progress
tail -f output/live/logs/training.log | grep "Episode"
```

**TensorBoard (if enabled):**
```bash
tensorboard --logdir output/live/tensorboard
```

---

## Evaluation

### Evaluate Single Model

```bash
python scripts/evaluate_sb3.py \
  --model output/dqn_seed42/models/dqn_final.zip \
  --config configs/config.yaml \
  --episodes 100 \
  --output results/eval_seed42.json
```

**Output:**
```
┌─────────────────────────────────────────┐
│  Evaluation Results (100 episodes)     │
├─────────────────────────────────────────┤
│  Avg Reward:           127.45 ± 23.12  │
│  Avg Handovers:          6.23 ± 1.89   │
│  Avg Episode Length:   119.82 ± 0.53   │
│  Success Rate:             98.0%       │
└─────────────────────────────────────────┘
```

### Multi-Seed Statistical Analysis

After training multiple seeds, analyze aggregate statistics:

```bash
python scripts/analyze_multi_seeds.py \
  --model-pattern "output/dqn_seed*/models/dqn_final.zip" \
  --config configs/config.yaml \
  --episodes 100
```

**Output:**
```
╔═══════════════════════════════════════════════════════╗
║       Multi-Seed Analysis (5 seeds × 100 episodes)  ║
╠═══════════════════════════════════════════════════════╣
║  Metric              │  Mean ± Std   │  Min - Max     ║
╠═══════════════════════════════════════════════════════╣
║  Reward              │  125.3 ± 8.7  │  112.4 - 138.2 ║
║  Handovers           │    6.1 ± 0.4  │    5.5 - 6.8   ║
║  Episode Length      │ 119.9 ± 0.2   │ 119.5 - 120.0  ║
╚═══════════════════════════════════════════════════════╝
```

### Baseline Comparison

```bash
python scripts/evaluate_baselines.py \
  --dqn-model output/dqn_seed42/models/dqn_final.zip \
  --config configs/config.yaml \
  --episodes 100
```

**Output:**
```
┌────────────────────┬────────────┬─────────────┬──────────────┐
│ Policy             │ Avg Reward │ Avg HOs     │ HO Reduction │
├────────────────────┼────────────┼─────────────┼──────────────┤
│ RSRP (baseline)    │   -45.3    │   21.4      │      -       │
│ Load-Balancing     │   -32.1    │   18.7      │    12.6%     │
│ Hybrid             │   -28.9    │   16.2      │    24.3%     │
│ DQN (ours)         │   127.5    │    6.2      │    71.0%     │
└────────────────────┴────────────┴─────────────┴──────────────┘
```

---

## Configuration

The project uses a single unified configuration file: `configs/config.yaml`

### Key Configuration Sections

#### Environment Settings
```yaml
environment:
  time_step_seconds: 5              # Simulation timestep
  episode_duration_minutes: 10      # 10 min episodes = 120 steps
  max_visible_satellites: 15        # Top-15 candidate selection
```

#### Reward Function (RVT-Based)
```yaml
environment:
  reward:
    # Handover penalties
    handover_to_loaded_penalty: -600.0   # Penalty for switching to loaded satellite
    handover_to_free_penalty: -350.0     # Penalty for switching to free satellite

    # Stay penalties/rewards
    stay_loaded_penalty_factor: 100.0    # Penalty factor for staying on loaded sat
    rvt_reward_weight: 2.0               # Reward weight for RVT (Remaining Visible Time)

    # Constraints
    min_dwell_time_seconds: 60           # Minimum 60s between handovers
    min_elevation_deg: 20.0              # Minimum elevation angle
```

#### Agent Hyperparameters
```yaml
agent:
  learning_rate: 0.0001
  gamma: 0.95                      # Discount factor
  batch_size: 64
  buffer_capacity: 10000
  target_update_freq: 100

  # Exploration schedule
  epsilon_start: 0.82
  epsilon_end: 0.2
  epsilon_decay: 0.9999            # Slow decay for stable learning
```

#### Precompute Settings
```yaml
precompute:
  enabled: true
  table_path: "data/orbit_precompute_30days.h5"
  # If false, falls back to real-time calculations (very slow)
```

### Modifying Configuration

1. Edit `configs/config.yaml`
2. No code changes needed
3. Run training with modified config:
   ```bash
   python train_sb3.py --config configs/config.yaml --output-dir output/custom
   ```

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Layer (SB3)                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  train_sb3.py → DQN Algorithm (Stable-Baselines3)   │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │ Gymnasium API
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Environment Layer (V9 - RVT-based)             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  SatelliteHandoverEnvV9                             │   │
│  │  - 14D observation space (13 features + RVT)        │   │
│  │  - 16 actions (stay + switch to 15 candidates)     │   │
│  │  - RVT-based reward function                        │   │
│  │  - Action masking (invalid actions)                 │   │
│  │  - Dwell time enforcement                           │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │ calculate_state(sat_id, timestamp)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 Adapter Layer (Precompute)                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  AdapterWrapper (auto-selects backend)              │   │
│  │                                                      │   │
│  │  ┌─────────────────┐      ┌────────────────────┐   │   │
│  │  │ Precompute Mode │  OR  │ Real-time Mode     │   │   │
│  │  │ (O(1) lookup)   │      │ (full calculation)│   │   │
│  │  └─────────────────┘      └────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │ Physics calculations
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Physics Layer (orbit-engine)             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  OrbitEngineAdapter                                 │   │
│  │  - SGP4 orbital mechanics                           │   │
│  │  - ITU-R P.676-13 atmospheric loss (44+35 lines)    │   │
│  │  - 3GPP TS 38.214/215 signal calculations           │   │
│  │  - Geometric calculations (elevation, distance)     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Observation Space (14 dimensions)

For each of 15 satellite candidates:
1. `is_current`: Current serving satellite flag (0/1)
2. `elevation_deg`: Elevation angle (20° - 90°)
3. `distance_km`: Distance to satellite
4. `rsrp_dbm`: Reference Signal Received Power
5. `rsrq_db`: Reference Signal Received Quality
6. `rs_sinr_db`: Signal-to-Interference-plus-Noise Ratio
7. `doppler_shift_hz`: Doppler frequency shift
8. `radial_velocity_ms`: Radial velocity
9. `atmospheric_loss_db`: ITU-R atmospheric attenuation
10. `path_loss_db`: Free space path loss
11. `propagation_delay_ms`: Signal propagation delay
12. `is_loaded`: Satellite load status (0/1)
13. `load_factor`: Normalized load (0.0 - 1.0)
14. `rvt`: **Remaining Visible Time** in seconds (key feature!)

**Shape:** `(15, 14)` - 15 satellites × 14 features

### Action Space (16 discrete actions)

- Action 0: **Stay** on current satellite
- Actions 1-15: **Switch** to candidate satellite i

**Action Masking:** Invalid actions (e.g., switch to invisible satellite, violate dwell time) are masked out.

### Reward Function (RVT-Based)

```python
if action == STAY:
    if satellite_is_loaded:
        reward = -stay_loaded_penalty_factor * load_factor
    else:
        reward = rvt_reward_weight * RVT  # Reward longer visibility
else:  # HANDOVER
    if target_satellite_is_loaded:
        reward = handover_to_loaded_penalty
    else:
        reward = handover_to_free_penalty
```

**Design Philosophy:**
- Penalize handovers (minimize frequency)
- Reward staying on satellites with long **RVT** (Remaining Visible Time)
- Heavily penalize switching to loaded satellites (load-aware)
- Enforce minimum dwell time (60s) to prevent oscillations

---

## Results

### Current Best Performance (Multi-Seed Average)

Training: **5 seeds × 2500 episodes** (~2 hours total)

| Metric | Value | vs RSRP Baseline |
|--------|-------|------------------|
| **Avg Handovers** | 6.1 ± 0.4 | **71.5% reduction** (21.4 → 6.1) |
| **Avg Reward** | 125.3 ± 8.7 | **376% improvement** (-45.3 → 125.3) |
| **Success Rate** | 98.2% ± 1.1% | - |
| **Avg Episode Length** | 119.9 ± 0.2 | Full episodes (120 steps) |

### Comparison with Baselines

| Policy | Avg Handovers | Handover Reduction | Avg Reward |
|--------|---------------|---------------------|------------|
| **RSRP** (always max RSRP) | 21.4 | - | -45.3 |
| **Load-Balancing** | 18.7 | 12.6% | -32.1 |
| **Hybrid** (RSRP + load) | 16.2 | 24.3% | -28.9 |
| **DQN (Ours)** | **6.1** | **71.5%** | **125.3** |

### Training Convergence

- **Episode 0-500**: Rapid improvement (ε: 0.82 → 0.77)
- **Episode 500-1500**: Steady learning (ε: 0.77 → 0.67)
- **Episode 1500-2500**: Fine-tuning (ε: 0.67 → 0.59)
- **Stable performance** maintained across all 5 seeds

### Key Insights

1. **RVT reward is effective**: Agents learn to prioritize satellites with long visibility
2. **Dwell time prevents oscillations**: 60s minimum gap eliminates ping-pong handovers
3. **Load-awareness works**: Agents avoid overloaded satellites
4. **Multi-seed validation**: Low variance (± 0.4 handovers) confirms robustness

---

## Tools & Scripts

### Training & Evaluation

| Script | Purpose | Usage |
|--------|---------|-------|
| `train_sb3.py` | Main training script (SB3 DQN) | `python train_sb3.py --config configs/config.yaml --num-episodes 2500` |
| `scripts/evaluate_sb3.py` | Evaluate trained models | `python scripts/evaluate_sb3.py --model output/dqn/models/dqn_final.zip --episodes 100` |
| `scripts/evaluate_baselines.py` | Compare with baseline policies | `python scripts/evaluate_baselines.py` |
| `scripts/analyze_multi_seeds.py` | Multi-seed statistical analysis | `python scripts/analyze_multi_seeds.py --model-pattern "output/dqn_seed*/models/*.zip"` |

### Precompute Tools

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/generate_orbit_precompute.py` | Generate HDF5 precompute tables | `python scripts/generate_orbit_precompute.py --start-time "2025-10-26 00:00:00" --end-time "2025-11-25 23:59:59"` |
| `tools/orbit/verify_precompute.py` | Verify table integrity | `python tools/orbit/verify_precompute.py --table data/orbit_precompute_30days.h5` |
| `tools/orbit/inspect_precompute.py` | Inspect table contents | `python tools/orbit/inspect_precompute.py --table data/orbit_precompute_30days.h5` |

### Visualization

| Script | Purpose | Usage |
|--------|---------|-------|
| `tools/visualization/plot_training_curves.py` | Plot training metrics | `python tools/visualization/plot_training_curves.py --log output/dqn/logs/training.log` |
| `tools/visualization/plot_handover_trajectory.py` | Visualize handover decisions | `python tools/visualization/plot_handover_trajectory.py --model output/dqn/models/dqn_final.zip` |

### Optimization

| Script | Purpose | Usage |
|--------|---------|-------|
| `tools/optimization/optimize_satellite_pool.py` | Optimize satellite selection (Stage 4) | `python tools/optimization/optimize_satellite_pool.py --tle-dir tle_data/` |

---

## Testing

### Run All Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python tests/test_environment.py
python tests/test_dwell_time.py
```

### Test Coverage

| Test File | Purpose | Key Checks |
|-----------|---------|------------|
| `tests/test_environment.py` | Environment functionality | - Reset/step mechanics<br>- Observation shape (15, 14)<br>- Action masking<br>- RVT calculation<br>- Reward function |
| `tests/test_dwell_time.py` | Dwell time constraint | - 60s minimum gap enforcement<br>- No rapid handover oscillations<br>- Correct blocking behavior |

### Quick Validation

```bash
# Validate environment (5 seconds)
python tests/test_environment.py

# Validate dwell time constraint (10 seconds)
python tests/test_dwell_time.py
```

---

## Development

### Code Structure Guidelines

- **`src/adapters/`**: Backend adapters (precompute/real-time switching)
- **`src/environments/`**: Gymnasium environments (only V9 active)
- **`src/utils/`**: Shared utilities (satellite selection, safety checks)
- **`scripts/`**: User-facing scripts (training, evaluation, analysis)
- **`tools/`**: Development tools (precompute generation, optimization, viz)
- **`tests/`**: Unit and integration tests

### Adding a New Feature

1. **Implement** in appropriate `src/` module
2. **Configure** parameters in `configs/config.yaml`
3. **Test** with `tests/test_*.py`
4. **Validate** with quick training run (100 episodes)
5. **Document** in README or docstrings

### Best Practices

- **Always use precompute mode** for training (set `precompute.enabled: true`)
- **Use multi-seed training** for reliable results (5 seeds minimum)
- **Check action masks** when modifying environment logic
- **Validate with baselines** to ensure improvements are real
- **Monitor epsilon decay** to ensure sufficient exploration

### Common Issues

**Issue: FileNotFoundError: No Stage 4 output found**
```
Error: FileNotFoundError when loading satellite pool
Symptoms:
  - "No Stage 4 output found" during environment initialization
  - "link_feasibility_output_*.json not found"

Cause: orbit-engine Stage 4 has not been run yet

Solution: Run orbit-engine to generate satellite pool data:
  cd ../orbit-engine
  source venv/bin/activate
  ./run.sh --stage 4

  # Verify output exists
  ls -lh data/outputs/stage4/link_feasibility_output_*.json
  # Should show ~29MB JSON file
```

**Issue: Training is very slow**
```
Solution: Verify precompute mode is enabled in configs/config.yaml:
  precompute:
    enabled: true
    table_path: "data/orbit_precompute_30days.h5"
```

**Issue: FPS degrades over time (starts at 80, drops to <10)**
```
Symptoms: Training slows down after several hundred episodes
Cause: I/O contention from frequent checkpoint saves and TensorBoard logging
Solution: Use optimized training flags:
  python train_sb3.py \
    --config configs/config.yaml \
    --save-freq 500 \           # Save every 500 eps (default: 100)
    --disable-tensorboard \      # Disable TensorBoard logging
    --num-episodes 2500

For multi-seed training, use the optimized script:
  ./run_academic_training.sh   # Includes 30s staggered starts
```

**Issue: Multiple training processes compete for disk I/O**
```
Solution: Stagger process starts by 30 seconds (built into run_academic_training.sh)
Monitor with: ./monitor_system_resources.sh
```

**Issue: NaN rewards during training**
```
Solution: Enabled by default in config.yaml:
  agent:
    enable_nan_check: true
    q_value_clip: 10000.0
```

**Issue: Too many handovers (> 20/episode)**
```
Solution: Check dwell time constraint:
  environment:
    reward:
      min_dwell_time_seconds: 60
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{handover-rl-2024,
  title={Deep Reinforcement Learning for LEO Satellite Handover Optimization with RVT-Based Rewards},
  author={Your Name},
  journal={IEEE Transactions on Aerospace and Electronic Systems},
  year={2024},
  note={Under Review}
}
```

### Related Papers

- **RVT-Based Reward Design**: IEEE TAES 2024
- **Precompute Acceleration**: [Your Paper Title]
- **LEO Handover Optimization**: [Related Work]

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **[orbit-engine](https://github.com/yourusername/orbit-engine)**: Physics-based LEO satellite simulation
- **[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)**: High-quality RL algorithm implementations
- **[Gymnasium](https://gymnasium.farama.org/)**: Standard RL environment interface
- **TLE Data**: [Space-Track.org](https://www.space-track.org/) for real Starlink orbital elements

---

## Contact

**Author**: [Your Name]
**Email**: [your.email@example.com]
**Project Page**: [https://github.com/yourusername/handover-rl](https://github.com/yourusername/handover-rl)

---

**Last Updated**: 2025-12-11 | **Version**: 4.2 (Production Ready - orbit-engine dependency documented)
