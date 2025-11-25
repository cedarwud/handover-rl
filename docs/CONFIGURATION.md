# Configuration Guide

**Purpose**: Complete guide to all configuration parameters
**Version**: 3.1
**Date**: 2025-11-25

---

## 📁 Configuration Files Overview

```
configs/
├── diagnostic_config.yaml        # Main training config (⭐ Most commonly used)
├── diagnostic_config_1day_test.yaml  # 1-day test config
├── diagnostic_config_realtime.yaml   # Real-time mode (no precompute)
└── strategies/                   # Baseline strategy configs
    ├── a4_based.yaml
    ├── d2_based.yaml
    └── strongest_rsrp.yaml
```

---

## ⚙️ Main Configuration: `diagnostic_config.yaml`

### 1. Precompute Settings (Top Priority)

```yaml
precompute:
  enabled: true  # Enable precompute mode (100x speedup)
  table_path: "data/orbit_precompute_30days_optimized.h5"
```

**Parameters**:
- **`enabled`** (boolean, default: `true`)
  - `true`: Use precompute table (recommended, 100x faster)
  - `false`: Real-time calculation (slow, only for debugging)

- **`table_path`** (string)
  - Path to HDF5 precompute table
  - Current: `data/orbit_precompute_30days_optimized.h5` (2025-10-26 to 2025-11-25)
  - Must match your generated table

**When to change**:
- After generating a new precompute table
- When switching between different time ranges

---

### 2. Environment Settings

```yaml
environment:
  time_step_seconds: 5
  episode_duration_minutes: 20
  max_visible_satellites: 10
  reward:
    qos_weight: 1.0
    sinr_weight: 0.3
    latency_weight: -0.2
    handover_penalty: -0.5
    ping_pong_penalty: -1.0
```

**Parameters**:

#### Time Configuration
- **`time_step_seconds`** (int, default: `5`)
  - Simulation time step
  - Smaller = finer resolution but slower training
  - Recommended: 5 seconds (balance of accuracy and speed)

- **`episode_duration_minutes`** (int, default: `20`)
  - Length of each training episode
  - Typical user session: 20 minutes
  - Longer = more data per episode but slower convergence
  - **Must match** precompute table time step!

#### Environment Constraints
- **`max_visible_satellites`** (int, default: `10`)
  - Maximum satellites in observation space
  - Matches 3GPP NR specifications
  - **Do not change** unless you know what you're doing

#### Reward Weights
- **`qos_weight`** (float, default: `1.0`)
  - Weight for QoS (Quality of Service) reward
  - Based on RSRP signal strength
  - Higher = prioritize signal quality

- **`sinr_weight`** (float, default: `0.3`)
  - Weight for SINR (Signal-to-Interference-plus-Noise Ratio)
  - Additional signal quality metric
  - Keep lower than qos_weight

- **`latency_weight`** (float, default: `-0.2`)
  - Weight for propagation delay
  - Negative = penalize high latency
  - LEO satellites have ~20-40ms latency

- **`handover_penalty`** (float, default: `-0.5`)
  - Penalty for each handover
  - Negative = discourage unnecessary handovers
  - **Key hyperparameter**: Adjust to control handover frequency

- **`ping_pong_penalty`** (float, default: `-1.0`)
  - Penalty for ping-pong handovers (switching back to previous satellite)
  - Should be more negative than handover_penalty
  - Prevents oscillation

**When to adjust**:
- **Reduce handovers**: Increase `handover_penalty` (e.g., -1.0)
- **Improve signal quality**: Increase `qos_weight` (e.g., 2.0)
- **Reduce ping-pong**: Increase `ping_pong_penalty` (e.g., -2.0)

---

### 3. Agent Configuration (DQN)

```yaml
agent:
  # Learning parameters
  learning_rate: 2.0e-5  # Conservative for stability
  gamma: 0.99            # Discount factor

  # Training parameters
  batch_size: 64
  buffer_capacity: 10000
  target_update_freq: 1000

  # Network architecture
  hidden_dim: 128

  # Exploration (epsilon-greedy)
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay: 0.995

  # Stability features
  enable_nan_check: true
  q_value_clip: 100.0
  use_huber_loss: true
```

**Core Parameters**:

#### Learning Rate
- **`learning_rate`** (float, default: `2.0e-5`)
  - How fast the agent learns
  - Too high = unstable training (loss explosion)
  - Too low = slow convergence
  - Current: Very conservative (2e-5) for stability
  - **Typical range**: 1e-5 to 1e-3

**When to adjust**:
- **Learning too slow**: Increase to 5e-5 or 1e-4
- **Loss exploding**: Decrease to 1e-5
- **After convergence**: Decrease for fine-tuning

#### Discount Factor
- **`gamma`** (float, default: `0.99`)
  - How much to value future rewards
  - 0 = only immediate reward
  - 1 = all future rewards equally important
  - 0.99 = standard for RL
  - **Do not change** unless you understand implications

#### Batch and Buffer
- **`batch_size`** (int, default: `64`)
  - Number of samples per training step
  - Larger = more stable but slower
  - Must be ≤ buffer_capacity
  - **Typical range**: 32-128

- **`buffer_capacity`** (int, default: `10000`)
  - Experience replay buffer size
  - Stores (state, action, reward, next_state) tuples
  - Larger = more diverse samples but more memory
  - **Typical range**: 10,000-100,000

- **`target_update_freq`** (int, default: `1000`)
  - How often to update target network (in steps)
  - Target network stabilizes learning
  - Larger = more stable but slower adaptation
  - **Typical range**: 100-10,000

#### Network Architecture
- **`hidden_dim`** (int, default: `128`)
  - Size of hidden layers in neural network
  - Larger = more capacity but slower training
  - **Typical range**: 64-512

#### Exploration (Epsilon-Greedy)
- **`epsilon_start`** (float, default: `1.0`)
  - Initial exploration rate
  - 1.0 = 100% random actions (pure exploration)
  - Start with full exploration to discover good states

- **`epsilon_end`** (float, default: `0.05`)
  - Final exploration rate after decay
  - 0.05 = 5% random actions (mostly exploitation)
  - Never go to 0 (always keep some exploration)

- **`epsilon_decay`** (float, default: `0.995`)
  - Epsilon multiplier per episode
  - epsilon = epsilon × decay
  - 0.995 = slow decay (~1400 episodes to reach epsilon_end)
  - 0.99 = faster decay (~300 episodes)
  - **Formula**: episodes_to_end ≈ log(epsilon_end/epsilon_start) / log(decay)

#### Stability Features (Added in v3.0)
- **`enable_nan_check`** (boolean, default: `true`)
  - Detect NaN/Inf in Q-values, rewards, states
  - **Keep enabled** for production

- **`q_value_clip`** (float, default: `100.0`)
  - Clip Q-values to [-clip, +clip]
  - Prevents explosion
  - **Keep enabled** for stability

- **`use_huber_loss`** (boolean, default: `true`)
  - Use Huber loss instead of MSE
  - More robust to outliers
  - **Keep enabled** for stability

---

### 4. Data Generation Settings

```yaml
data_generation:
  time_span_days: 30
  time_step_seconds: 5
  episode_duration_minutes: 20

  satellites:
    total: 97  # Changed from 101
    starlink: 97
    oneweb: 0
    source: "orbit-engine Stage 4 Pool Optimization (Starlink constellation)"

  tle_strategy:
    method: "multi_tle_daily"
    tle_files_count: 30
    propagation_per_tle_days: 1
    tle_directory: "../tle_data/starlink/tle"
```

**Parameters**:
- **`time_span_days`** (int): Total time range for data generation
- **`satellites.total`** (int): Number of satellites (97 from orbit-engine)
- **`tle_strategy.method`** (string): TLE selection strategy
  - `"multi_tle_daily"`: Use multiple TLE files, each for 1 day
  - Ensures <1km accuracy

**Note**: These are metadata for precompute generation. Once table is generated, these values are fixed.

---

## 🎛️ Training Level Configuration

Defined in `src/configs/training_levels.py`:

```python
TRAINING_LEVELS = {
    0: {'name': 'Smoke Test', 'num_episodes': 10},
    1: {'name': 'Quick Validation', 'num_episodes': 50},  # ⭐ Recommended
    2: {'name': 'Development', 'num_episodes': 200},
    3: {'name': 'Validation', 'num_episodes': 500},
    4: {'name': 'Baseline', 'num_episodes': 1000},
    5: {'name': 'Full Training', 'num_episodes': 1700},  # Publication
    6: {'name': 'Long-term', 'num_episodes': 4174},     # 1M+ steps
}
```

**Usage**:
```bash
python train.py --algorithm dqn --level 1  # Quick Validation
```

**When to use each level**:
- **Level 0**: System verification (1-2 min)
- **Level 1**: Hyperparameter testing (5-10 min) ⭐
- **Level 2**: Development iteration (20-40 min)
- **Level 3**: Paper draft experiments (1-1.5 hours)
- **Level 4**: Baseline experiments (2-3 hours)
- **Level 5**: Publication quality (3-5 hours)
- **Level 6**: Long-term convergence (8-10 hours)

---

## 🔧 Common Configuration Scenarios

### Scenario 1: Reduce Handover Frequency

**Problem**: Too many handovers, want more stable connections

**Solution**: Increase handover penalty

```yaml
environment:
  reward:
    handover_penalty: -1.0  # Increased from -0.5
    ping_pong_penalty: -2.0  # Also increase
```

### Scenario 2: Improve Signal Quality

**Problem**: RSRP too low, want better signal

**Solution**: Increase QoS weight

```yaml
environment:
  reward:
    qos_weight: 2.0  # Increased from 1.0
    handover_penalty: -0.3  # Reduce penalty to allow more handovers
```

### Scenario 3: Speed Up Learning

**Problem**: Training converges too slowly

**Solution 1**: Increase learning rate (try first)
```yaml
agent:
  learning_rate: 5.0e-5  # Increased from 2e-5
```

**Solution 2**: Faster epsilon decay
```yaml
agent:
  epsilon_decay: 0.99  # Decreased from 0.995
```

**Solution 3**: Smaller target update frequency
```yaml
agent:
  target_update_freq: 500  # Decreased from 1000
```

### Scenario 4: Stabilize Training

**Problem**: Loss exploding or NaN values

**Solution 1**: Decrease learning rate
```yaml
agent:
  learning_rate: 1.0e-5  # Decreased from 2e-5
```

**Solution 2**: Enable all stability features (should already be on)
```yaml
agent:
  enable_nan_check: true
  q_value_clip: 100.0
  use_huber_loss: true
```

**Solution 3**: Increase target update frequency
```yaml
agent:
  target_update_freq: 2000  # Increased from 1000
```

### Scenario 5: Generate Different Time Range

**Problem**: Need precompute table for different dates

**Step 1**: Generate new table
```bash
python scripts/generate_orbit_precompute.py \
  --start-time "2025-12-01 00:00:00" \
  --end-time "2025-12-31 23:59:59" \
  --output data/orbit_precompute_december.h5 \
  --config configs/diagnostic_config.yaml \
  --processes 16 \
  --yes
```

**Step 2**: Update config
```yaml
precompute:
  enabled: true
  table_path: "data/orbit_precompute_december.h5"
```

**Step 3**: Train (time range auto-detected!)
```bash
python train.py --algorithm dqn --level 5 --config configs/diagnostic_config.yaml
```

---

## 📚 Advanced Topics

### Custom Reward Function

To modify reward calculation, edit `src/environments/satellite_handover_env.py`:

```python
def _calculate_reward(self, action, prev_satellite):
    # Current reward calculation
    qos_reward = self.config['reward']['qos_weight'] * normalized_rsrp
    handover_penalty = self.config['reward']['handover_penalty']

    # Add your custom reward components here
    # Example: Add distance-based reward
    distance_reward = -0.1 * (current_distance / 1000)  # Penalize far satellites

    total_reward = qos_reward + handover_penalty + distance_reward
    return total_reward
```

### Multiple Algorithms

Current: DQN and Double DQN

To add new algorithm (e.g., PPO):
1. Implement in `src/agents/ppo/`
2. Register in `train.py` ALGORITHM_REGISTRY
3. Use with `--algorithm ppo`

---

## ✅ Configuration Checklist

Before starting training:

- [ ] Precompute table generated
- [ ] `precompute.enabled = true` in config
- [ ] `precompute.table_path` points to correct file
- [ ] Training level selected (recommend Level 1 first)
- [ ] Reward weights appropriate for your objective
- [ ] Learning rate appropriate (start with 2e-5)
- [ ] Output directory specified

---

## 🔗 Related Documentation

- **[README.md](../README.md)** - Project overview
- **[PRECOMPUTE_QUICKSTART.md](PRECOMPUTE_QUICKSTART.md)** - Generate precompute table
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Training guide
- **[FAQ.md](FAQ.md)** - Common questions
- **[CHANGELOG.md](../CHANGELOG.md)** - Version history

---

**Questions?** Check [FAQ.md](FAQ.md) or open an issue on GitHub.
