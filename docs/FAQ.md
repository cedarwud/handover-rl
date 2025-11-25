# Frequently Asked Questions (FAQ)

**Version**: 3.1
**Date**: 2025-11-25

---

## 🚀 Getting Started

### Q1: What's the fastest way to start training?

**A**: Follow these 3 steps:

```bash
# 1. Generate 30-day precompute table (~30 minutes)
python scripts/generate_orbit_precompute.py \
  --start-time "2025-10-26 00:00:00" \
  --end-time "2025-11-25 23:59:59" \
  --output data/orbit_precompute_30days_optimized.h5 \
  --config configs/diagnostic_config.yaml \
  --processes 16 \
  --yes

# 2. Verify config points to table (should already be set)
# Check configs/diagnostic_config.yaml: precompute.enabled = true

# 3. Run Level 0 smoke test (1-2 minutes)
python train.py --algorithm dqn --level 0 --output-dir output/smoke_test
```

See [PRECOMPUTE_QUICKSTART.md](PRECOMPUTE_QUICKSTART.md) for details.

---

### Q2: Do I need a GPU?

**A**: No, but it helps:

- **CPU only**: Works fine! Most computation is in environment simulation, not neural network
- **With GPU**: Training ~20-30% faster, especially for larger networks
- **Recommendation**: Start with CPU, add GPU if you're doing Level 5+ training frequently

The precompute system already provides 100x speedup, so GPU is optional.

---

### Q3: How much disk space and RAM do I need?

**Disk Space**:
- Precompute table (30 days): ~2.5 GB
- Precompute table (7 days): ~563 MB
- Checkpoints: ~10-50 MB per training run
- Logs: ~1-5 MB per training run
- **Recommendation**: 10 GB free space

**RAM**:
- Training: ~2-4 GB
- Precompute generation: ~4-8 GB (with 16 processes)
- **Recommendation**: 8 GB RAM minimum, 16 GB preferred

---

## 📊 Precompute Tables

### Q4: How long does it take to generate a precompute table?

**A**: With optimized parallel mode (16 processes):

| Duration | Generation Time | File Size |
|----------|----------------|-----------|
| 1 day    | ~1 minute      | ~85 MB    |
| 7 days   | ~7 minutes     | ~563 MB   |
| 14 days  | ~15 minutes    | ~1.1 GB   |
| 30 days  | ~30 minutes    | ~2.5 GB   |

**Note**: First time may take longer due to TLE file downloads.

---

### Q5: Which precompute table should I use?

**A**: Depends on your training level:

- **Level 0-1** (quick tests): 7-day table is sufficient
- **Level 2-4** (development): 14-day table recommended
- **Level 5-6** (production/publication): 30-day table required

**Pro Tip**: Use 30-day table for all levels - it works for everything and avoids regeneration.

---

### Q6: Can I use dates in the past or future?

**A**: Yes! The dates are just labels. The physics simulation uses:

- **Orbital mechanics**: Based on TLE data (valid for ~1-2 weeks from TLE epoch)
- **Signal propagation**: Physics-based (ITU-R, 3GPP models)
- **No dependency on "current time"**

**Recommendation**: Use any 30-day range. The system automatically detects and uses the table's time range during training.

---

### Q7: What if I get "Timestamp out of range" error?

**Error**:
```
ValueError: Timestamp 2025-12-01 out of range.
Table range: 2025-10-26 to 2025-11-25
```

**Solutions**:

1. **Automatic (Recommended)**: Training now auto-detects time range from table metadata. This error should not occur in v3.1+.

2. **Manual**: If you still see this error, regenerate the precompute table covering the needed time range:
```bash
python scripts/generate_orbit_precompute.py \
  --start-time "2025-11-01 00:00:00" \
  --end-time "2025-12-01 00:00:00" \
  --output data/orbit_precompute_november.h5 \
  --config configs/diagnostic_config.yaml \
  --processes 16 \
  --yes
```

3. **Update config** to point to new table:
```yaml
precompute:
  table_path: "data/orbit_precompute_november.h5"
```

---

### Q8: Why is my precompute generation slow?

**A**: Check these issues:

1. **Not using optimized parallel mode?**
   - Add `--processes 16` flag
   - Expected speed: ~1.73M points/minute
   - If you see "Falling back to standard parallel mode", TLE preloading failed

2. **Low number of processes?**
   - Use 8-16 processes for best performance
   - More than 16 has diminishing returns

3. **Slow disk?**
   - HDF5 writes are disk-intensive
   - Use SSD if possible

**Verification**:
```bash
# Should see this message:
✅ Optimized parallel computation succeeded!
   Preloaded TLE data for 97 satellites
```

---

## 🎯 Training Configuration

### Q9: How do I adjust the learning rate?

**A**: Edit `configs/diagnostic_config.yaml`:

```yaml
agent:
  learning_rate: 2.0e-5  # Default (very conservative)
```

**Guidelines**:
- **Too slow convergence**: Increase to 5e-5 or 1e-4
- **Loss exploding/NaN**: Decrease to 1e-5 or 5e-6
- **Fine-tuning**: Decrease after initial convergence

**Quick test**: Run Level 1 (50 episodes) to validate new learning rate.

See [CONFIGURATION.md](CONFIGURATION.md#learning-rate) for details.

---

### Q10: How do I reduce handover frequency?

**A**: Increase handover penalty:

```yaml
environment:
  reward:
    handover_penalty: -1.0  # Increased from -0.5
    ping_pong_penalty: -2.0  # Also increase proportionally
```

**Trade-off**: Higher penalty = fewer handovers but potentially lower signal quality.

**Verification**: Check TensorBoard metrics for "handovers_per_episode" - should decrease.

---

### Q11: How do I improve signal quality (RSRP)?

**A**: Increase QoS weight:

```yaml
environment:
  reward:
    qos_weight: 2.0  # Increased from 1.0
    handover_penalty: -0.3  # Optionally reduce to allow more handovers
```

**Trade-off**: Higher QoS weight = better signal but more handovers.

**Verification**: Check TensorBoard metrics for "mean_rsrp" - should increase.

---

### Q12: What training level should I use?

**A**: Depends on your goal:

| Level | Use Case | Time | When to Use |
|-------|----------|------|-------------|
| **0** | Smoke Test | 1-2 min | First time, verify setup |
| **1** | Quick Validation | 5-10 min | Testing hyperparameters, daily development |
| **2** | Development | 20-40 min | Iterating on algorithms |
| **3** | Validation | 1-1.5 hours | Paper draft, proving concept |
| **4** | Baseline | 2-3 hours | Establishing baselines |
| **5** | Full Training | 3-5 hours | Paper experiments |
| **6** | Long-term | 28-34 hours | Publication (1M+ steps) |

**Recommendation**:
- Start with **Level 1** for all development
- Use **Level 5** for paper results
- Use **Level 6** for final publication

---

### Q13: Can I change the episode duration?

**A**: Yes, but requires regenerating precompute table:

```yaml
environment:
  episode_duration_minutes: 30  # Changed from 20
```

**Important**: Also update in config used for precompute generation to ensure timesteps match.

**When to change**:
- Studying longer user sessions
- Testing convergence over extended periods

---

### Q14: How do I switch to Double DQN?

**A**: Use `--algorithm ddqn`:

```bash
python train.py --algorithm ddqn --level 1 --output-dir output/ddqn_test
```

**Difference**:
- **DQN**: Uses same network for action selection and evaluation (can overestimate)
- **Double DQN**: Separates selection and evaluation (more stable, slightly slower)

**Recommendation**: Try both at Level 1, compare results.

---

## 🐛 Troubleshooting

### Q15: Training is not converging - what should I check?

**A**: Systematic debugging:

1. **Check if training is running**:
   - Look for increasing episode numbers
   - Monitor TensorBoard

2. **Check for NaN/Inf**:
   ```bash
   grep -i "nan\|inf" output/*/training.log
   ```
   - If found: Decrease learning rate or increase gradient clipping

3. **Check epsilon decay**:
   - Early training should be mostly exploration (epsilon near 1.0)
   - If epsilon drops too fast, agent doesn't explore enough
   - Solution: Increase `epsilon_decay` (e.g., 0.995 → 0.997)

4. **Check reward scale**:
   - Rewards should be in range [-10, +10] typically
   - If rewards are extreme, adjust reward weights

5. **Verify environment**:
   - Check that satellites are visible
   - Check that RSRP values are reasonable (-140 to -80 dBm)

**Quick validation**:
```bash
python train.py --algorithm dqn --level 0 --output-dir output/debug
```

---

### Q16: My training stopped unexpectedly - what happened?

**A**: Check these common causes:

1. **Out of memory**:
   ```bash
   dmesg | grep -i "out of memory"
   ```
   - Solution: Reduce `buffer_capacity` or `batch_size`

2. **Disk full**:
   ```bash
   df -h
   ```
   - Solution: Clean up old checkpoints/logs

3. **Process killed**:
   ```bash
   grep -i "killed" output/*/training.log
   ```
   - Often due to OOM killer

4. **Timestamp out of range** (v3.0 and earlier):
   - Fixed in v3.1 with automatic time range detection
   - If still occurring, regenerate precompute table

**Resume training**: Load checkpoint and continue:
```bash
python train.py \
  --algorithm dqn \
  --level 5 \
  --resume output/level5_full/checkpoints/checkpoint_epoch_800.pth \
  --output-dir output/level5_full
```

---

### Q17: TensorBoard shows no data - why?

**A**: Common issues:

1. **Wrong directory**:
   ```bash
   tensorboard --logdir output/  # Use parent directory, not specific run
   ```

2. **Training just started**:
   - Wait for first episode to complete
   - Refresh browser (Ctrl+R)

3. **Log writer not flushed**:
   - Data writes every N steps
   - Be patient or check for errors in training log

4. **Port conflict**:
   - Default port 6006 might be in use
   - Try: `tensorboard --logdir output/ --port 6007`

---

### Q18: How do I compare multiple training runs?

**A**: TensorBoard can show multiple runs:

```bash
# Organize runs with clear names
python train.py --algorithm dqn --level 5 --output-dir output/dqn_lr2e5
python train.py --algorithm dqn --level 5 --output-dir output/dqn_lr5e5

# View all in TensorBoard
tensorboard --logdir output/
```

TensorBoard will automatically group runs by directory name.

**Pro Tip**: Use descriptive output directory names that indicate the experiment (e.g., `output/dqn_lr2e5_penalty1.0`).

---

## 🔬 Advanced Topics

### Q19: Can I add a new baseline algorithm?

**A**: Yes! Follow these steps:

1. **Implement in `src/agents/your_algorithm/`**:
   ```
   src/agents/your_algorithm/
   ├── __init__.py
   ├── agent.py  # Must have train() and select_action() methods
   └── network.py
   ```

2. **Register in `train.py`**:
   ```python
   ALGORITHM_REGISTRY = {
       'dqn': dqn_agent.DQNAgent,
       'ddqn': ddqn_agent.DoubleDQNAgent,
       'your_algo': your_algorithm.YourAgent,  # Add here
   }
   ```

3. **Test**:
   ```bash
   python train.py --algorithm your_algo --level 0
   ```

See existing DQN/DDQN implementations as templates.

---

### Q20: How do I modify the reward function?

**A**: Edit `src/environments/satellite_handover_env.py`:

```python
def _calculate_reward(self, action, prev_satellite):
    # Current reward components
    qos_reward = self.config['reward']['qos_weight'] * normalized_rsrp
    sinr_reward = self.config['reward']['sinr_weight'] * normalized_sinr
    latency_penalty = self.config['reward']['latency_weight'] * normalized_latency

    # Add your custom component
    # Example: Penalize switching to distant satellites
    if action != 0:  # If switching
        new_sat = self.visible_satellites[action - 1]
        distance_penalty = -0.1 * (new_sat.distance / 1000)  # Penalize per 1000 km
    else:
        distance_penalty = 0

    total_reward = (qos_reward + sinr_reward +
                    latency_penalty + handover_penalty +
                    ping_pong_penalty + distance_penalty)

    return total_reward
```

**Important**: Document your changes for reproducibility!

---

### Q21: How do I use a custom satellite constellation?

**A**: This requires modifying orbit-engine configuration:

1. **Add TLE files** to `../tle_data/your_constellation/`

2. **Update config**:
   ```yaml
   data_generation:
     satellites:
       total: <your_number>
       starlink: 0
       your_constellation: <your_number>
     tle_strategy:
       tle_directory: "../tle_data/your_constellation/tle"
   ```

3. **Regenerate precompute table** with new constellation

**Note**: This is advanced usage. See orbit-engine documentation for details.

---

## 📚 Best Practices

### Q22: What's the recommended workflow for a new experiment?

**A**: Follow this workflow:

```bash
# 1. Quick validation (5-10 min)
python train.py --algorithm dqn --level 1 --output-dir output/exp1_test

# 2. Check results look reasonable
tensorboard --logdir output/

# 3. If good, run full training (3-5 hours)
python train.py --algorithm dqn --level 5 --output-dir output/exp1_full

# 4. Evaluate
python evaluate.py \
  --model output/exp1_full/checkpoints/best_model.pth \
  --algorithm dqn \
  --episodes 50

# 5. Compare with baselines
python scripts/compare_baselines.py --models output/exp1_full/...
```

**Key principle**: Always validate with Level 1 before investing time in Level 5+.

---

### Q23: How often should I save checkpoints?

**A**: Default settings are reasonable:

```python
TRAINING_LEVELS = {
    1: {'checkpoint_interval': 10},  # Save every 10 episodes
    5: {'checkpoint_interval': 50},  # Save every 50 episodes
}
```

**Adjust if**:
- Disk space limited: Increase interval
- Frequent crashes: Decrease interval
- Long training: Keep more frequent checkpoints early

**Pro Tip**: The system always saves `best_model.pth` based on validation reward.

---

### Q24: Should I use the same random seed for reproducibility?

**A**: For research papers, yes:

```bash
python train.py \
  --algorithm dqn \
  --level 5 \
  --seed 42 \
  --output-dir output/exp1_seed42
```

**Recommendation for paper**:
- Run 3-5 seeds (e.g., 42, 123, 456, 789, 2024)
- Report mean ± std across seeds
- This demonstrates statistical significance

---

## 🎓 Academic and Publication

### Q25: What should I cite in my paper?

**A**: Key references:

1. **Deep Q-Learning**: Mnih et al., "Playing Atari with Deep Reinforcement Learning" (2013)
2. **Double DQN**: van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (2015)
3. **3GPP Standards**: 3GPP TS 38.214/215 (5G NR Physical Layer Procedures)
4. **ITU-R Model**: ITU-R P.676-13 (Atmospheric attenuation)
5. **SGP4**: Vallado, "Fundamentals of Astrodynamics and Applications" (2013)

For this project specifically, cite the technical reports in `docs/` if made public.

---

### Q26: How do I explain the precompute system in my paper?

**A**: Example text:

> "To accelerate training, we employ a precomputed orbit state table. All physics-based calculations (ITU-R P.676-13 atmospheric propagation, 3GPP TS 38.214/215 signal modeling, and SGP4 orbital mechanics) are performed offline using real TLE data from Space-Track.org. During training, the agent performs O(1) table lookups instead of real-time calculations, achieving a 100× speedup without sacrificing physical accuracy. The precompute approach is methodologically sound as it separates environment physics (deterministic) from agent learning (stochastic), ensuring reproducibility while dramatically reducing computational cost."

**Key points to emphasize**:
- No loss of physical accuracy
- Deterministic environment
- Reproducible results
- Standard practice in RL (similar to Gym environments)

---

### Q27: What metrics should I report in my paper?

**A**: Essential metrics:

**Performance**:
- Mean episode reward (with std across seeds)
- Mean RSRP (dBm)
- Handover rate (handovers per episode)
- Ping-pong rate (%)

**Learning**:
- Training time (wall-clock hours)
- Convergence episode number
- Sample efficiency (reward vs. timesteps)

**Comparison**:
- Performance vs. baselines (A4-based, D2-based, Strongest-RSRP)
- Improvement percentage
- Statistical significance (t-test, p-value)

**Example table**:
```
| Method | Mean Reward | RSRP (dBm) | Handovers | Improvement |
|--------|-------------|------------|-----------|-------------|
| A4     | 120 ± 5     | -95.2      | 12.3      | Baseline    |
| DQN    | 180 ± 8     | -88.5      | 3.8       | +50% ⭐     |
```

---

## 🔗 Related Documentation

- **[README.md](../README.md)** - Project overview
- **[PRECOMPUTE_QUICKSTART.md](PRECOMPUTE_QUICKSTART.md)** - Generate precompute table
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Training guide
- **[CONFIGURATION.md](CONFIGURATION.md)** - Configuration parameters
- **[CHANGELOG.md](../CHANGELOG.md)** - Version history

---

**Still have questions?** Open an issue on GitHub or check the documentation in `docs/`.
