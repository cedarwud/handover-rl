#!/bin/bash
# Test 30-core training with full episode length

cd /home/sat/satellite/handover-rl
source venv/bin/activate

python3 train.py \
  --algorithm dqn \
  --level 1 \
  --output-dir output/test_full_episodes_30cores \
  --num-envs 30 \
  --seed 42 \
  2>&1 | tee test_full_episodes_30cores.log
