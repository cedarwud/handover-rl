#!/bin/bash
# Test single seed training with optimizations
# Use this to verify fixes before running all 5 seeds

SEED=42
EPISODES=100  # Short test: 100 episodes (~30 minutes)

echo "=========================================="
echo "Single Seed Test Training (Optimized)"
echo "=========================================="
echo "Seed: $SEED"
echo "Episodes: $EPISODES (test run)"
echo "Checkpoint frequency: every 500 episodes (no checkpoint in test)"
echo "TensorBoard: disabled"
echo "Output: output/test_seed${SEED}"
echo "=========================================="
echo ""

# Clean up previous test
rm -rf output/test_seed${SEED}
rm -f /tmp/test_seed${SEED}.log

echo "[$(date +%H:%M:%S)] Starting test training..."

venv/bin/python train_sb3.py \
  --config configs/config.yaml \
  --output-dir output/test_seed${SEED} \
  --num-episodes $EPISODES \
  --seed $SEED \
  --save-freq 500 \
  --disable-tensorboard \
  2>&1 | tee /tmp/test_seed${SEED}.log

echo ""
echo "=========================================="
echo "Test training completed!"
echo "=========================================="
echo ""
echo "Check FPS performance:"
grep "| *fps" /tmp/test_seed${SEED}.log | awk '{print $3}' | sed 's/|//g' | tail -20
echo ""
echo "Final results:"
tail -20 /tmp/test_seed${SEED}.log
