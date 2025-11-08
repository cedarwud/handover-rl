#!/bin/bash
# 訓練完成後的分析腳本（現在可以準備，訓練完成後執行）

LOG_FILE="training_level5_20min_final.log"
OUTPUT_DIR="output/level5_20min_final"

echo "========================================"
echo "訓練結果分析"
echo "========================================"
echo ""

# 1. 訓練統計
echo "=== 訓練統計 ==="
TOTAL_EPISODES=$(grep -c "Training:.*1700" $LOG_FILE)
echo "完成 episodes: $TOTAL_EPISODES"

TRAINING_STEPS=$(grep "Training steps:" $LOG_FILE | tail -1 | grep -oP '\d+')
echo "總訓練 steps: $TRAINING_STEPS"

START_TIME=$(head -100 $LOG_FILE | grep "Starting training" -A 1 | tail -1)
END_TIME=$(tail -100 $LOG_FILE | grep "Training completed" -A 1 | tail -1)
echo "開始時間: $START_TIME"
echo "結束時間: $END_TIME"
echo ""

# 2. Episode 920 檢查
echo "=== Episode 920 Bug 檢查 ==="
EP920_DATA=$(grep "Episode.*920.*reward=" $LOG_FILE)
if [ -n "$EP920_DATA" ]; then
    echo "$EP920_DATA"

    EP920_LOSS=$(echo "$EP920_DATA" | grep -oP 'loss=\K[0-9.]+')
    if (( $(echo "$EP920_LOSS < 10" | bc -l) )); then
        echo "✅ Episode 920 loss 正常: $EP920_LOSS"
    else
        echo "⚠️  Episode 920 loss 異常: $EP920_LOSS"
    fi
else
    echo "❌ 未找到 Episode 920 數據"
fi
echo ""

# 3. 數值穩定性
echo "=== 數值穩定性 ==="
NAN_COUNT=$(grep -c "\[NaN" $LOG_FILE)
INF_COUNT=$(grep -c "\[Inf" $LOG_FILE)
LARGE_LOSS=$(grep -c "Large Loss" $LOG_FILE)

echo "NaN 警告: $NAN_COUNT 次"
echo "Inf 警告: $INF_COUNT 次"
echo "Large Loss: $LARGE_LOSS 次"

if [ $((NAN_COUNT + INF_COUNT + LARGE_LOSS)) -eq 0 ]; then
    echo "✅ 完全穩定，無數值問題"
else
    echo "⚠️  發現數值問題"
fi
echo ""

# 4. 性能分析
echo "=== 性能趨勢 ==="
echo "前 100 episodes 平均 reward:"
grep "Episode.*reward=" $LOG_FILE | head -10 | grep -oP 'reward=\K-?[0-9.]+' | \
    awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}'

echo "後 100 episodes 平均 reward:"
grep "Episode.*reward=" $LOG_FILE | tail -10 | grep -oP 'reward=\K-?[0-9.]+' | \
    awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}'
echo ""

# 5. Checkpoint 檢查
echo "=== Checkpoint 檢查 ==="
if [ -d "$OUTPUT_DIR/checkpoints" ]; then
    CKPT_COUNT=$(ls $OUTPUT_DIR/checkpoints/*.pth 2>/dev/null | wc -l)
    echo "保存的 checkpoints: $CKPT_COUNT 個"

    if [ $CKPT_COUNT -gt 0 ]; then
        echo "最新 checkpoint:"
        ls -lht $OUTPUT_DIR/checkpoints/*.pth | head -3
    fi
else
    echo "❌ Checkpoint 目錄不存在"
fi
echo ""

# 6. 建議
echo "=== 後續建議 ==="
echo "1. 使用 TensorBoard 查看詳細曲線:"
echo "   tensorboard --logdir=$OUTPUT_DIR/logs"
echo ""
echo "2. 評估最終模型:"
echo "   python evaluate.py --checkpoint $OUTPUT_DIR/checkpoints/best_model.pth"
echo ""
echo "3. 與 baseline 對比:"
echo "   python compare_models.py"
echo ""

echo "========================================"
