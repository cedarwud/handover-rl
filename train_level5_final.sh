#!/bin/bash
# Level 5 完整訓練 - 最終版本
# Episode: 20 分鐘（240 steps）
# Total: 1700 episodes × 240 steps = 408,000 steps
# 預計時間: ~85 小時（3.5 天）

cd /home/sat/satellite/handover-rl
source venv/bin/activate

echo "=========================================="
echo "Level 5 完整訓練啟動"
echo "=========================================="
echo "配置："
echo "  Algorithm: DQN"
echo "  Episodes: 1700"
echo "  Episode 長度: 20 分鐘（240 steps）"
echo "  總訓練量: 408,000 steps"
echo "  並行環境: 30"
echo "  預計時間: ~85 小時"
echo "=========================================="
echo ""

python3 train.py \
  --algorithm dqn \
  --level 5 \
  --output-dir output/level5_20min_final \
  --config config/diagnostic_config.yaml \
  --num-envs 30 \
  --seed 42 \
  2>&1 | tee training_level5_20min_final.log

echo ""
echo "=========================================="
echo "訓練完成！"
echo "=========================================="
