#!/bin/bash
# Test training with RAM disk precompute (2 concurrent seeds)
# Quick validation: 100 episodes per seed

SEEDS=(42 123)
MAX_CONCURRENT=2

echo "=========================================="
echo "RAM Disk Test (2 seeds × 100 episodes)"
echo "=========================================="
echo "Testing: RAM disk I/O performance"
echo "Seeds: ${SEEDS[@]}"
echo "Episodes: 100 (quick test)"
echo "Precompute: /dev/shm/orbit_precompute_30days.h5"
echo "=========================================="
echo ""

# Verify precompute is in RAM disk
if [ ! -f "/dev/shm/orbit_precompute_30days.h5" ]; then
    echo "❌ Error: Precompute file not found in /dev/shm"
    exit 1
fi

echo "✓ Precompute file verified in RAM disk"
echo ""

# Clean old test data
rm -rf output/test_seed* /tmp/test_seed*.log

# Start 2 seeds concurrently
for seed in "${SEEDS[@]}"; do
    echo "[$(date +%H:%M:%S)] Starting test with seed $seed..."

    venv/bin/python train_sb3.py \
        --config configs/config.yaml \
        --output-dir output/test_seed${seed} \
        --num-episodes 100 \
        --seed $seed \
        --save-freq 500 \
        --disable-tensorboard \
        2>&1 | tee /tmp/test_seed${seed}.log &

    PID=$!
    echo "  → Seed $seed started (PID: $PID)"
    echo ""
    sleep 5
done

echo "=========================================="
echo "Test training started!"
echo "=========================================="
echo ""
echo "Monitor FPS:"
echo "  watch -n 5 'grep \"| *fps\" /tmp/test_seed*.log | tail -10'"
echo ""

# Monitor until completion
echo "Monitoring (press Ctrl+C to stop monitoring)..."
while [ $(pgrep -f "train_sb3.py.*test_seed" | wc -l) -gt 0 ]; do
    sleep 10
    for seed in "${SEEDS[@]}"; do
        LATEST_FPS=$(grep "| *fps" /tmp/test_seed${seed}.log 2>/dev/null | tail -1 | awk '{print $3}' | sed 's/|//g')
        LATEST_TS=$(grep "total_timesteps" /tmp/test_seed${seed}.log 2>/dev/null | tail -1 | awk '{print $3}' | sed 's/|//g')
        if [ ! -z "$LATEST_FPS" ]; then
            echo "[$(date +%H:%M:%S)] Seed $seed: FPS=$LATEST_FPS, Timesteps=$LATEST_TS/12000"
        fi
    done
    echo "---"
done

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
echo ""

# Check FPS stability
for seed in "${SEEDS[@]}"; do
    echo "Seed $seed FPS history:"
    grep "| *fps" /tmp/test_seed${seed}.log | awk '{print $3}' | sed 's/|//g' | tail -20
    echo ""
done

echo "If FPS remains stable (70-80), RAM disk solution is working!"
