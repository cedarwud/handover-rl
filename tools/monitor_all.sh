#!/bin/bash
# Monitor All Background Tasks
# Usage: ./tools/monitor_all.sh

clear
echo "========================================"
echo "🚀 Handover-RL 並行任務監控"
echo "========================================"
echo "時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 1. Level 2 Training
echo "📊 Level 2 Training (200 episodes)"
echo "----------------------------------------"
if [ -f "/tmp/level2_training.log" ]; then
    EPISODES=$(grep -c "Episode [0-9]\+/" /tmp/level2_training.log 2>/dev/null || echo "0")
    echo "✅ 已完成 episodes: $EPISODES / 200"

    # Latest progress
    tail -100 /tmp/level2_training.log 2>/dev/null | grep -E "Training:|Episode [0-9]+/" | tail -3

    # Check if finished
    if [ -f "output/level2_precompute/checkpoints/final_model.pth" ]; then
        FINISH_TIME=$(stat -c %y output/level2_precompute/checkpoints/final_model.pth | cut -d'.' -f1)
        echo "✅ 完成時間: $FINISH_TIME"
    fi
else
    echo "❌ 訓練日誌不存在"
fi

echo ""
echo "========================================"

# 2. 30-day Precompute Generation (Latest: 2025-10-10 to 2025-11-08)
echo "💾 30-day 預計算表生成 (最新 TLE 數據)"
echo "----------------------------------------"
LOG_FILE="/tmp/precompute_30day_latest.log"
if [ -f "$LOG_FILE" ]; then
    # Time range
    echo "📅 時間範圍: 2025-10-10 to 2025-11-08 (29 天)"

    # Get satellite progress
    PROGRESS=$(tail -100 "$LOG_FILE" 2>/dev/null | grep "Satellites:" | tail -1)
    if [ -n "$PROGRESS" ]; then
        echo "進度: $PROGRESS"
    fi

    # File size
    if [ -f "data/orbit_precompute_30days.h5" ]; then
        SIZE=$(ls -lh data/orbit_precompute_30days.h5 | awk '{print $5}')
        echo "✅ 當前大小: $SIZE (預計 ~1.4 GB)"

        # Estimated completion (assuming ~3.5 hours = 210 min)
        START_TIME=$(stat -c %W data/orbit_precompute_30days.h5)
        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - START_TIME))
        ELAPSED_MIN=$((ELAPSED / 60))
        REMAINING=$((210 - ELAPSED_MIN))
        [ $REMAINING -lt 0 ] && REMAINING=0
        echo "⏱️  已執行: ${ELAPSED_MIN} 分鐘"
        echo "⏱️  預計剩餘: ~${REMAINING} 分鐘 (總計 ~210 分鐘)"
    else
        echo "⏳ 文件尚未創建 (workers 初始化中)"
    fi
else
    echo "❌ 預計算日誌不存在: $LOG_FILE"
fi

echo ""
echo "========================================"

# 3. System Resources
echo "🖥️  系統資源使用"
echo "----------------------------------------"
echo "CPU 負載: $(uptime | awk -F'load average:' '{print $2}')"
echo "記憶體: $(free -h | grep Mem | awk '{print $3 " / " $2}')"

# GPU if available
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
    GPU_UTIL=$(echo $GPU_INFO | cut -d',' -f1)
    GPU_MEM=$(echo $GPU_INFO | cut -d',' -f2,3 | sed 's/,/ \/ /')
    echo "GPU 使用率: ${GPU_UTIL}%"
    echo "GPU 記憶體: ${GPU_MEM} MB"
fi

echo ""
echo "========================================"
echo "💡 使用方式:"
echo "   watch -n 30 ./tools/monitor_all.sh  # 每30秒更新"
echo "   ./tools/monitor_all.sh              # 單次查看"
echo "========================================"
