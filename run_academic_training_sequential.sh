#!/bin/bash
# Academic Standard Multi-Seed Training (Sequential batches to avoid I/O contention)
# Strategy: Run 2-3 seeds at a time to reduce I/O load
#
# Optimizations:
# - Precompute table in RAM disk (/dev/shm)
# - Checkpoint save frequency: 500 episodes
# - TensorBoard disabled
# - Max 2 concurrent seeds to avoid I/O contention

SEEDS=(42 123 456 789 2024)
MAX_CONCURRENT=2

echo "=========================================="
echo "Academic Training (Sequential Batches)"
echo "=========================================="
echo "Configuration: configs/config.yaml"
echo "Seeds: ${SEEDS[@]}"
echo "Episodes per seed: 2500"
echo "Max concurrent: $MAX_CONCURRENT"
echo "Precompute: /dev/shm/orbit_precompute_30days.h5 (RAM disk)"
echo "Output: output/academic_seed*"
echo "=========================================="
echo ""

# Verify precompute is in RAM disk
if [ ! -f "/dev/shm/orbit_precompute_30days.h5" ]; then
    echo "❌ Error: Precompute file not found in /dev/shm"
    echo "Run: cp data/orbit_precompute_30days.h5 /dev/shm/"
    exit 1
fi

echo "✓ Precompute file verified in RAM disk"
echo ""

# Function to wait for running processes
wait_for_slots() {
    while [ $(pgrep -f "train_sb3.py.*academic_seed" | wc -l) -ge $MAX_CONCURRENT ]; do
        echo "  [$(date +%H:%M:%S)] Waiting for training slot... ($(pgrep -f 'train_sb3.py.*academic_seed' | wc -l)/$MAX_CONCURRENT running)"
        sleep 30
    done
}

# Start training for each seed
for seed in "${SEEDS[@]}"; do
    # Wait if we have too many concurrent processes
    wait_for_slots

    echo "[$(date +%H:%M:%S)] Starting training with seed $seed..."

    venv/bin/python train_sb3.py \
        --config configs/config.yaml \
        --output-dir output/academic_seed${seed} \
        --num-episodes 2500 \
        --seed $seed \
        --save-freq 500 \
        --disable-tensorboard \
        2>&1 | tee /tmp/academic_seed${seed}.log &

    PID=$!
    echo "  → Seed $seed training started (PID: $PID)"
    echo "  → Log: /tmp/academic_seed${seed}.log"
    echo ""

    # Brief pause before starting next
    sleep 10
done

echo "=========================================="
echo "All seeds queued!"
echo "=========================================="
echo ""
echo "Monitor progress:"
echo "  watch -n 10 'pgrep -f train_sb3.py.*academic | wc -l'"
echo ""
echo "Check logs:"
echo "  tail -f /tmp/academic_seed42.log"
echo ""

# Wait for all to complete
echo "Waiting for all training to complete..."
while [ $(pgrep -f "train_sb3.py.*academic_seed" | wc -l) -gt 0 ]; do
    RUNNING=$(pgrep -f "train_sb3.py.*academic_seed" | wc -l)
    echo "[$(date +%H:%M:%S)] Still running: $RUNNING seeds"
    sleep 60
done

echo ""
echo "=========================================="
echo "✅ All training completed!"
echo "=========================================="

# Show results
echo ""
echo "Results:"
for seed in "${SEEDS[@]}"; do
    if [ -f "output/academic_seed${seed}/models/dqn_final.zip" ]; then
        echo "  Seed $seed: ✅ Completed"
    else
        PROGRESS=$(grep "total_timesteps" /tmp/academic_seed${seed}.log 2>/dev/null | tail -1 | awk '{print $3}' | sed 's/|//g')
        echo "  Seed $seed: ⚠️  Incomplete ($PROGRESS/300000)"
    fi
done
