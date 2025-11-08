#!/bin/bash
# 查看監控日誌

MONITOR_LOG="training_monitor.log"

if [ ! -f "$MONITOR_LOG" ]; then
    echo "❌ 監控日誌尚未生成"
    echo "監控會在啟動後 5 分鐘進行第一次檢查"
    exit 1
fi

echo "=========================================="
echo "📊 智能監控狀態"
echo "=========================================="
echo ""

# 檢查監控進程
if ps aux | grep -q "[a]uto_monitor.sh"; then
    echo "✅ 監控進程：運行中"
    MONITOR_PID=$(ps aux | grep "[a]uto_monitor.sh" | awk '{print $2}')
    echo "   PID: $MONITOR_PID"
else
    echo "❌ 監控進程：未運行"
fi
echo ""

# 顯示最新的檢查結果
echo "📝 最近的檢查記錄:"
echo "=========================================="
tail -50 "$MONITOR_LOG"
echo "=========================================="
echo ""

echo "💡 提示："
echo "  - 監控每 5 分鐘檢查一次"
echo "  - 持續監控: tail -f training_monitor.log"
echo "  - 停止監控: pkill -f auto_monitor.sh"
