#!/bin/bash
# Full Training Script - 2500 episodes × 5 seeds
# Academic Standard: Multi-seed experiments for statistical significance

SEEDS=(42 123 456 789 2024)
CONFIG="configs/config.yaml"
BASE_OUTPUT="output/academic_experiment_20251217"

echo "======================================"
echo "開始完整訓練實驗"
echo "時間: $(date)"
echo "Seeds: ${SEEDS[@]}"
echo "Episodes: 2500"
echo "預估總時間: ~5 小時"
echo "======================================"
echo ""

for seed in "${SEEDS[@]}"; do
    OUTPUT_DIR="${BASE_OUTPUT}/seed_${seed}"
    echo ""
    echo "======================================"
    echo "訓練 Seed ${seed}"
    echo "開始時間: $(date)"
    echo "輸出目錄: ${OUTPUT_DIR}"
    echo "======================================"
    
    /home/sat/satellite/handover-rl/venv/bin/python train_sb3.py \
        --config "${CONFIG}" \
        --output-dir "${OUTPUT_DIR}" \
        --num-episodes 2500 \
        --seed ${seed}
    
    echo ""
    echo "Seed ${seed} 完成時間: $(date)"
    echo ""
done

echo ""
echo "======================================"
echo "所有訓練完成！"
echo "完成時間: $(date)"
echo "======================================"
