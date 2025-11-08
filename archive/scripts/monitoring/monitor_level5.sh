#!/bin/bash
# Monitor Level 5 Training Progress

LOG_FILE="training_level5_20min_final.log"
PID=1052344

echo "========================================"
echo "Level 5 訓練監控"
echo "========================================"
date
echo ""

# Check process
if ps -p $PID > /dev/null 2>&1; then
    echo "✅ 訓練進程運行中 (PID: $PID)"
    ELAPSED=$(ps -p $PID -o etime= | xargs)
    echo "   運行時間: $ELAPSED"
else
    echo "❌ 訓練進程已停止"
fi

echo ""
echo "=== 訓練進度 ==="
grep "Training:" $LOG_FILE | tail -5

echo ""
echo "=== Episode 完成情況 ==="
COMPLETED=$(grep "Training:" $LOG_FILE | tail -1 | grep -oP '\d+/1700' | cut -d'/' -f1)
if [ -n "$COMPLETED" ]; then
    PERCENT=$(echo "scale=1; $COMPLETED * 100 / 1700" | bc)
    echo "已完成: $COMPLETED / 1700 episodes ($PERCENT%)"

    # Estimate remaining time
    grep "Training:" $LOG_FILE | tail -1 | grep -oP '<[^>]+' | sed 's/<//g'
fi

echo ""
echo "=== Log 文件大小 ==="
ls -lh $LOG_FILE | awk '{print $5, $9}'

echo ""
echo "=== 最新輸出（過濾 WARNING）==="
tail -20 $LOG_FILE | grep -v "WARNING:environments"

echo ""
echo "========================================"
echo "提示:"
echo "  - 實時監控: tail -f $LOG_FILE"
echo "  - 自動更新: watch -n 60 ./monitor_level5.sh"
echo "  - 總預計時間: ~79 小時"
echo "========================================"
