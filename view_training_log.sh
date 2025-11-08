#!/bin/bash
# 查看訓練日誌 - 過濾重要信息

echo "========================================"
echo "訓練日誌查看器"
echo "========================================"
echo ""

LOG_FILE="training_level5_20min_final.log"

echo "📊 最新 Episode 統計 (每 10 episodes 報告):"
echo "----------------------------------------"
grep "Episode.*reward=" "$LOG_FILE" | tail -5
echo ""

echo "📈 最新訓練進度:"
echo "----------------------------------------"
grep "Training:" "$LOG_FILE" | grep -v "WARNING" | tail -1
echo ""

echo "📝 日誌統計:"
echo "----------------------------------------"
TOTAL_LINES=$(wc -l < "$LOG_FILE")
EPISODE_LINES=$(grep -c "Episode.*reward=" "$LOG_FILE")
WARNING_LINES=$(grep -c "WARNING" "$LOG_FILE")

echo "  總行數: $TOTAL_LINES"
echo "  Episode 報告: $EPISODE_LINES 個"
echo "  警告數: $WARNING_LINES"
echo ""

echo "💡 持續監控:"
echo "  tail -f training_level5_20min_final.log | grep --line-buffered 'Episode.*reward='"
echo ""
echo "📄 完整日誌:"
echo "  less training_level5_20min_final.log"
