# Episode 524 Stuck Bug - Investigation Report

## Bug Summary

**Status**: Under Investigation
**Severity**: Critical - Blocks Level 4 training completion
**First Observed**: November 13, 2025

### Symptom

Training consistently gets stuck around Episode 524-525, regardless of restarts:

- **Normal episode time**: ~24 seconds/episode
- **Stuck episode time**: 200+ seconds (hangs indefinitely)
- **System impact**: High CPU load (18-30), system becomes unresponsive
- **Recovery**: Requires process kill and restart

### Occurrence History

1. **First occurrence**: Episode 525 (previous training session)
2. **Second occurrence**: Episode 524 (November 13, 23:39)
3. **Pattern**: Happens at 52-53% progress in Level 4 (1000 episodes)

## Investigation Findings

### ✅ What We've Ruled Out

1. **Time range issue**: Episode 524 accesses timestamps from 2025-10-13 15:20-15:40, which is well within the 29-day precompute table (2025-10-10 to 2025-11-08)

2. **Data corruption**: Initial HDF5 checks show no obvious NaN/Inf in the precompute table at that time range

3. **Replay buffer overflow**: Buffer fills after ~42 episodes, but stuck occurs at Episode 524 (12x later)

4. **Checkpoint timing**: Episode 524 is 24 episodes after checkpoint 500 (checkpoints save every 100 episodes)

### 🔍 Suspicious Patterns

1. **Episode numbers**:
   - Episode 512 = 2^9 (power of 2 - might trigger optimization edge case)
   - Episode 524 = 2² × 131
   - Both in the 510-530 range

2. **Timing**:
   - Episode 524 corresponds to Day 3.64 (2025-10-13 15:20)
   - This is the afternoon of the 4th day of simulation

3. **System behavior**:
   - CPU load spikes dramatically
   - Episode processing time jumps from 24s to 200+s
   - System becomes unresponsive (suggests I/O or computation bottleneck)

### 💡 Hypotheses

1. **HDF5 I/O bottleneck**: Accessing specific time ranges might trigger slow reads
2. **Memory leak accumulation**: Python garbage collection delayed until Episode ~524
3. **Numerical instability**: Accumulated floating-point errors in DQN weights
4. **Cache invalidation**: System runs out of HDF5 chunk cache at this point
5. **PyTorch CUDA synchronization**: GPU synchronization issue after certain number of batches

## Current Setup

### Monitoring Scripts

1. **Auto-restart monitor** (`monitor_and_restart.sh`):
   - Detects stuck condition (>5 minutes without progress)
   - Auto-restarts from latest checkpoint
   - **Issue**: Episode counter resets to 0 on resume (checkpoint doesn't save episode number)

2. **Episode 524 diagnostic** (`diagnostic_episode524.sh` - NEW):
   - Activates when training reaches Episode 510
   - Captures detailed system state at: 510, 512, 515, 520, 522, 523, 524, 525, 526, 530
   - For Episode 524 specifically: monitors every 10 seconds with full diagnostics
   - Logs to: `/tmp/episode524_diagnostic.log`

### Diagnostic Data to Capture

When Episode 524 is reached (ETA: ~4 hours from now), the diagnostic script will capture:

- Process memory usage (RSS, VSZ)
- CPU utilization and time
- System load average
- Disk I/O statistics
- GPU usage (if available)
- Training log tail
- Timing: How long Episode 524 takes vs Episode 522/523

## Current Training Status

**As of November 14, 02:12 UTC**:
- Episode: 359/1000 (36%)
- Speed: ~24 seconds/episode
- ETA to Episode 524: ~4 hours
- Both monitoring scripts active and ready

## Next Steps

### Immediate Actions

1. **Wait for Episode 524**: Diagnostic script will capture detailed data when training reaches Episode 524
2. **Analyze diagnostic logs**: Check `/tmp/episode524_diagnostic.log` for patterns
3. **Compare Episode 522 vs 524**: Look for differences in system state

### Possible Fixes (After Diagnosis)

1. **If HDF5 I/O issue**:
   - Adjust HDF5 chunk cache settings
   - Pre-load data for episodes 520-530
   - Enable HDF5 SWMR (Single Writer Multiple Reader) mode

2. **If memory leak**:
   - Add explicit garbage collection after every N episodes
   - Reduce replay buffer size
   - Clear PyTorch cache periodically

3. **If numerical instability**:
   - Add gradient norm logging
   - Check for NaN/Inf in Q-values at Episode 524
   - Increase gradient clipping threshold

4. **If checkpoint issue**:
   - Fix checkpoint to save episode counter
   - Add episode skip logic to bypass Episode 524 temporarily

## Files Modified/Created

- `/home/sat/satellite/handover-rl/monitor_and_restart.sh` (existing - auto-restart)
- `/home/sat/satellite/handover-rl/diagnostic_episode524.sh` (NEW - detailed diagnostics)
- `/home/sat/satellite/handover-rl/EPISODE524_BUG_REPORT.md` (this file)

## Log Files

- Training log: `/tmp/level4_training_monitored.log`
- Monitor log: `/tmp/training_monitor.log`
- Diagnostic log: `/tmp/episode524_diagnostic.log` (will populate when Episode 510 reached)

---

**Last Updated**: November 14, 2025 02:12 UTC
**Investigator**: Claude
**User Contact**: Requested manual review after diagnostic data collection
