#!/bin/bash
# Academic Standard Multi-Seed Training (Optimized for I/O)
# 5 seeds × 2500 episodes = Academic publication standard
#
# Optimizations:
# - Checkpoint save frequency: 500 episodes (reduced from 100)
# - TensorBoard disabled to reduce I/O overhead
# - Staggered start (30s delay between seeds) to avoid I/O contention

SEEDS=(42 123 456 789 2024)

echo "=========================================="
echo "Academic Standard Training (Optimized)"
echo "=========================================="
echo "Configuration: configs/config.yaml"
echo "Seeds: ${SEEDS[@]}"
echo "Episodes per seed: 2500"
echo "Checkpoint frequency: every 500 episodes"
echo "TensorBoard: disabled"
echo "Output: output/academic_seed*"
echo "=========================================="
echo ""

for seed in "${SEEDS[@]}"; do
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

  # Stagger starts by 30 seconds to reduce simultaneous I/O
  if [ "$seed" != "2024" ]; then
    echo "  ⏸  Waiting 30s before starting next seed (reduce I/O contention)..."
    sleep 30
  fi
done

echo "=========================================="
echo "All training processes started!"
echo "=========================================="
echo ""
echo "Monitor progress:"
echo "  ./monitor_academic_training.sh"
echo ""
echo "Check individual logs:"
echo "  tail -f /tmp/academic_seed42.log"
echo ""
echo "Active processes:"
pgrep -f "train_sb3.py.*academic_seed" -a
