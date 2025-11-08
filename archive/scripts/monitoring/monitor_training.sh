#!/bin/bash
# 訓練監控腳本 - 每10分鐘自動檢查一次

LOG_FILE="long_training_17k.log"
ALERT_FILE="training_alerts.txt"
CHECK_INTERVAL=600  # 10分鐘

echo "=== 訓練監控腳本啟動 ==="
echo "日誌文件: $LOG_FILE"
echo "檢查間隔: $CHECK_INTERVAL 秒 (10分鐘)"
echo "警報文件: $ALERT_FILE"
echo ""

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 檢查日誌文件是否存在
    if [ ! -f "$LOG_FILE" ]; then
        echo "[$TIMESTAMP] ⚠️  警告：日誌文件不存在，訓練可能尚未開始"
        sleep $CHECK_INTERVAL
        continue
    fi
    
    echo "[$TIMESTAMP] 開始檢查..."
    
    # 1. 檢查 NaN/Inf 錯誤
    NAN_COUNT=$(grep -c "NaN/Inf Detection" "$LOG_FILE" 2>/dev/null || echo "0")
    if [ "$NAN_COUNT" -gt 0 ]; then
        echo "[$TIMESTAMP] ❌ 發現 NaN/Inf 錯誤: $NAN_COUNT 次" | tee -a "$ALERT_FILE"
        echo "最新錯誤:" | tee -a "$ALERT_FILE"
        grep "NaN/Inf Detection" "$LOG_FILE" | tail -3 | tee -a "$ALERT_FILE"
    fi
    
    # 2. 檢查 Large Loss 警告
    LARGE_LOSS=$(grep -c "Large Loss Warning" "$LOG_FILE" 2>/dev/null || echo "0")
    if [ "$LARGE_LOSS" -gt 0 ]; then
        echo "[$TIMESTAMP] ⚠️  發現 Large Loss 警告: $LARGE_LOSS 次" | tee -a "$ALERT_FILE"
        grep "Large Loss Warning" "$LOG_FILE" | tail -3 | tee -a "$ALERT_FILE"
    fi
    
    # 3. 檢查進度
    CURRENT_EPISODE=$(grep "Episode" "$LOG_FILE" | tail -1 | grep -oP 'Episode\s+\K[0-9]+' || echo "0")
    TOTAL_EPISODES=17000
    if [ "$CURRENT_EPISODE" -gt 0 ]; then
        PROGRESS=$(echo "scale=2; $CURRENT_EPISODE * 100 / $TOTAL_EPISODES" | bc)
        echo "[$TIMESTAMP] 📊 進度: Episode $CURRENT_EPISODE/$TOTAL_EPISODES ($PROGRESS%)"
    fi
    
    # 4. 檢查最新 loss
    LATEST_LOSS=$(grep "loss=" "$LOG_FILE" | tail -1 | grep -oP 'loss=\K[0-9.e+-]+' || echo "N/A")
    echo "[$TIMESTAMP] 📉 最新 loss: $LATEST_LOSS"
    
    # 5. 檢查 GPU 記憶體
    if command -v nvidia-smi &> /dev/null; then
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        echo "[$TIMESTAMP] 🎮 GPU 記憶體: ${GPU_MEM}MB"
    fi
    
    # 6. 檢查訓練是否卡住（最後一行超過30分鐘沒更新）
    LAST_LINE_TIME=$(stat -c %Y "$LOG_FILE" 2>/dev/null || stat -f %m "$LOG_FILE" 2>/dev/null || echo "0")
    CURRENT_TIME=$(date +%s)
    TIME_DIFF=$((CURRENT_TIME - LAST_LINE_TIME))
    if [ "$TIME_DIFF" -gt 1800 ]; then  # 30分鐘
        echo "[$TIMESTAMP] ⚠️  警告：日誌文件 $TIME_DIFF 秒沒有更新（超過30分鐘）" | tee -a "$ALERT_FILE"
        echo "訓練可能已卡住或崩潰！" | tee -a "$ALERT_FILE"
    fi
    
    # 7. 檢查 Episode 920 附近
    if [ "$CURRENT_EPISODE" -ge 915 ] && [ "$CURRENT_EPISODE" -le 925 ]; then
        echo "[$TIMESTAMP] 🎯 關鍵區域：正在通過 Episode 920"
        grep "Episode  920" "$LOG_FILE" | tail -1
    fi
    
    echo "[$TIMESTAMP] ✅ 檢查完成"
    echo "---"
    
    sleep $CHECK_INTERVAL
done
