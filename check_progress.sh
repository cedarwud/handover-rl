#!/bin/bash
# 快速查看訓練進度

LOG_FILE="training_level5_20min_final.log"

echo "=========================================="
echo "🚀 訓練進度報告"
echo "=========================================="
echo ""

# 檢查訓練進程
if ps aux | grep -q "[p]ython.*train.py"; then
    echo "✅ 訓練進程：運行中"
    TRAIN_PID=$(ps aux | grep "[p]ython.*train.py" | head -1 | awk '{print $2}')
    echo "   主進程 PID: $TRAIN_PID"
else
    echo "❌ 訓練進程：未運行"
fi
echo ""

# 最新 Episodes
echo "📊 最新完成的 Episodes:"
grep "Episode.*reward=" "$LOG_FILE" 2>/dev/null | tail -5 || echo "   尚未完成任何 episode"
echo ""

# 統計
TOTAL_EPISODES=$(grep "Episode.*reward=" "$LOG_FILE" 2>/dev/null | wc -l)
INVALID_ACTIONS=$(grep "WARNING.*Action.*out of range" "$LOG_FILE" 2>/dev/null | wc -l)

echo "📈 統計資訊:"
echo "   完成 Episodes: $TOTAL_EPISODES / 1700 ($(awk "BEGIN {printf \"%.1f\", $TOTAL_EPISODES/1700*100}")%)"
echo "   無效動作警告: $INVALID_ACTIONS"

if [ "$INVALID_ACTIONS" -eq 0 ]; then
    echo "   ✅ Action Masking 正常運作"
else
    echo "   ⚠️  發現無效動作，請檢查"
fi
echo ""

# 日誌大小
LOG_SIZE=$(du -h "$LOG_FILE" 2>/dev/null | cut -f1)
echo "📝 日誌文件大小: $LOG_SIZE"
echo ""

# GPU 使用情況
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU 狀態:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F', ' '{printf "   GPU %s (%s): %s%% GPU, %.0f/%.0fMB 記憶體\n", $1, $2, $3, $4, $5}'
fi

echo ""
echo "=========================================="
