#!/bin/bash
# Monitor 30-core training progress

echo "=== 30 核心訓練監控 ==="
echo ""

# Check process
PID=1028246
if ps -p $PID > /dev/null; then
    echo "✅ 訓練進程運行中 (PID: $PID)"
    CPU_USAGE=$(ps -p $PID -o %cpu | tail -1)
    MEM_USAGE=$(ps -p $PID -o %mem | tail -1)
    ELAPSED=$(ps -p $PID -o etime | tail -1)
    echo "   CPU: ${CPU_USAGE}%  |  記憶體: ${MEM_USAGE}%  |  運行時間: $ELAPSED"
else
    echo "❌ 訓練進程已停止"
fi

echo ""
echo "=== 訓練進度 ==="
# Get latest progress
grep -E "Episode.*reward=" test_full_episodes_30cores.log | tail -5

echo ""
echo "=== 最新輸出 (最後 10 行) ==="
tail -10 test_full_episodes_30cores.log

echo ""
echo "=== Log 文件大小 ==="
wc -l test_full_episodes_30cores.log

echo ""
echo "提示: 運行 'watch -n 60 ./monitor_30cores.sh' 自動每分鐘更新"
