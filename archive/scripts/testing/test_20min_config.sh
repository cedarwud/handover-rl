#!/bin/bash
# 快速驗證 20 分鐘配置
# 50 episodes × 240 steps = 12,000 steps
# 預計時間: ~2.5 小時

cd /home/sat/satellite/handover-rl
source venv/bin/activate

echo "=========================================="
echo "20 分鐘配置驗證測試"
echo "=========================================="
echo "配置："
echo "  Episodes: 50"
echo "  Episode 長度: 20 分鐘（240 steps）"
echo "  預計每 episode: ~3 分鐘"
echo "  預計總時間: ~2.5 小時"
echo "=========================================="
echo ""

python3 train.py \
  --algorithm dqn \
  --level 1 \
  --output-dir output/test_20min_config \
  --config config/diagnostic_config.yaml \
  --num-envs 30 \
  --seed 42 \
  2>&1 | tee test_20min_config.log

echo ""
echo "=========================================="
echo "測試完成！檢查 episode 長度是否為 240 steps"
echo "=========================================="
