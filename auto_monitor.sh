#!/bin/bash
# 智能訓練監控 - 自動檢查並修復問題

LOG_FILE="training_level5_20min_final.log"
MONITOR_LOG="training_monitor.log"
CHECK_INTERVAL=300  # 5 分鐘檢查一次

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🤖 智能監控啟動" | tee -a "$MONITOR_LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 檢查間隔: ${CHECK_INTERVAL}秒 ($(($CHECK_INTERVAL/60))分鐘)" | tee -a "$MONITOR_LOG"
echo "" | tee -a "$MONITOR_LOG"

LAST_EPISODE_COUNT=0
STUCK_COUNT=0
LAST_CHECK_TIME=$(date +%s)

while true; do
    sleep $CHECK_INTERVAL

    CURRENT_TIME=$(date +%s)
    TIME_DIFF=$((CURRENT_TIME - LAST_CHECK_TIME))

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 定期檢查 ==========" | tee -a "$MONITOR_LOG"

    # 檢查 1: 訓練進程是否運行
    if ! ps aux | grep -q "[p]ython.*train.py"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ 嚴重: 訓練進程已停止！" | tee -a "$MONITOR_LOG"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 正在檢查日誌最後幾行..." | tee -a "$MONITOR_LOG"
        tail -20 "$LOG_FILE" | tee -a "$MONITOR_LOG"

        # 檢查是否是正常完成
        if grep -q "Training completed" "$LOG_FILE" 2>/dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 訓練已正常完成" | tee -a "$MONITOR_LOG"
            exit 0
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  訓練異常終止，需要人工檢查" | tee -a "$MONITOR_LOG"
            # 不自動重啟，等待用戶決定
            exit 1
        fi
    fi

    # 檢查 2: Episode 進度
    CURRENT_EPISODE_COUNT=$(grep "Episode.*reward=" "$LOG_FILE" 2>/dev/null | wc -l)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📊 完成 Episodes: $CURRENT_EPISODE_COUNT / 1700" | tee -a "$MONITOR_LOG"

    # 檢查是否卡住（長時間沒有新 episode）
    if [ "$CURRENT_EPISODE_COUNT" -eq "$LAST_EPISODE_COUNT" ]; then
        STUCK_COUNT=$((STUCK_COUNT + 1))

        # 只在卡住超過2次（10分鐘）才開始警告
        if [ "$STUCK_COUNT" -ge 2 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  注意: $(($TIME_DIFF * $STUCK_COUNT))秒內沒有新 episode (連續 $STUCK_COUNT 次)" | tee -a "$MONITOR_LOG"
        fi

        # 如果連續 6 次檢查都沒有進度（30分鐘），可能有問題
        if [ "$STUCK_COUNT" -ge 6 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ 嚴重: 30分鐘內沒有進度，可能卡住了" | tee -a "$MONITOR_LOG"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 檢查 GPU 使用情況..." | tee -a "$MONITOR_LOG"
            nvidia-smi | tee -a "$MONITOR_LOG"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  需要人工檢查，監控暫停" | tee -a "$MONITOR_LOG"
            exit 1
        fi
    else
        # 有進度，重置計數
        NEW_EPISODES=$((CURRENT_EPISODE_COUNT - LAST_EPISODE_COUNT))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 新完成 $NEW_EPISODES 個 episodes" | tee -a "$MONITOR_LOG"
        STUCK_COUNT=0
        LAST_EPISODE_COUNT=$CURRENT_EPISODE_COUNT

        # 顯示最新的 episode
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 最新 episode:" | tee -a "$MONITOR_LOG"
        grep "Episode.*reward=" "$LOG_FILE" | tail -1 | tee -a "$MONITOR_LOG"
    fi

    # 檢查 3: 無效動作警告
    INVALID_ACTIONS=$(grep "WARNING.*Action.*out of range" "$LOG_FILE" 2>/dev/null | wc -l)
    if [ "$INVALID_ACTIONS" -gt 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ 嚴重: 發現 $INVALID_ACTIONS 個無效動作警告！" | tee -a "$MONITOR_LOG"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Action Masking 沒有正常工作" | tee -a "$MONITOR_LOG"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  需要人工檢查代碼" | tee -a "$MONITOR_LOG"
        # 這個問題需要代碼修復，不能自動恢復
        exit 1
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Action Masking 正常運作 (0 個無效動作)" | tee -a "$MONITOR_LOG"
    fi

    # 檢查 4: Loss 是否異常
    if [ "$CURRENT_EPISODE_COUNT" -gt 0 ]; then
        LATEST_LOSS=$(grep "Episode.*reward=" "$LOG_FILE" | tail -1 | grep -oP "loss=\K[0-9.]+|loss=\K(nan|inf)" || echo "0")
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📉 最新 Loss: $LATEST_LOSS" | tee -a "$MONITOR_LOG"

        if [ "$LATEST_LOSS" = "nan" ] || [ "$LATEST_LOSS" = "inf" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ 嚴重: Loss 爆炸 (NaN/Inf)！" | tee -a "$MONITOR_LOG"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 這可能是數值穩定性問題" | tee -a "$MONITOR_LOG"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  需要人工檢查" | tee -a "$MONITOR_LOG"
            exit 1
        fi

        # 檢查 loss 是否異常大（> 1000）
        if command -v bc &> /dev/null; then
            if (( $(echo "$LATEST_LOSS > 1000" | bc -l) )); then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  警告: Loss 異常大 (>1000)" | tee -a "$MONITOR_LOG"
            fi
        fi
    fi

    # 檢查 5: GPU 使用情況
    if command -v nvidia-smi &> /dev/null; then
        GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🎮 GPU: ${GPU_UTIL}% 使用率, ${GPU_MEM}MB 記憶體" | tee -a "$MONITOR_LOG"

        # 註：GPU使用率為0%通常是正常的
        # 因為大部分時間在做環境步驟（CPU），DQN訓練非常快（毫秒級）
        # 只有當GPU記憶體也為0時才警告（表示模型可能沒載入）
        if [ "$GPU_MEM" -lt 100 ] && [ "$CURRENT_EPISODE_COUNT" -gt 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  警告: GPU 記憶體異常低，模型可能未載入到GPU" | tee -a "$MONITOR_LOG"
        fi
    fi

    # 檢查 6: 日誌文件大小
    LOG_SIZE=$(du -m "$LOG_FILE" 2>/dev/null | cut -f1)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📝 日誌大小: ${LOG_SIZE}MB" | tee -a "$MONITOR_LOG"

    # 如果日誌超過 1GB，警告
    if [ "$LOG_SIZE" -gt 1000 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  警告: 日誌文件超過 1GB" | tee -a "$MONITOR_LOG"
    fi

    # 進度預測
    if [ "$CURRENT_EPISODE_COUNT" -gt 0 ] && [ "$STUCK_COUNT" -eq 0 ]; then
        PROGRESS_PERCENT=$(awk "BEGIN {printf \"%.1f\", $CURRENT_EPISODE_COUNT/1700*100}")
        REMAINING_EPISODES=$((1700 - CURRENT_EPISODE_COUNT))

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📈 進度: ${PROGRESS_PERCENT}%, 剩餘 ${REMAINING_EPISODES} episodes" | tee -a "$MONITOR_LOG"

        # 里程碑提醒
        if [ "$CURRENT_EPISODE_COUNT" -eq 10 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🎉 達成里程碑: Episode 10" | tee -a "$MONITOR_LOG"
        elif [ "$CURRENT_EPISODE_COUNT" -eq 50 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🎉 達成里程碑: Episode 50" | tee -a "$MONITOR_LOG"
        elif [ "$CURRENT_EPISODE_COUNT" -eq 100 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🎉 達成里程碑: Episode 100" | tee -a "$MONITOR_LOG"
        elif [ "$CURRENT_EPISODE_COUNT" -eq 920 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🎉🎉🎉 達成關鍵里程碑: Episode 920！" | tee -a "$MONITOR_LOG"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⭐ 這是 bug 修復驗證點，請檢查 loss 是否 < 10" | tee -a "$MONITOR_LOG"
        fi
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 檢查完成 ==========\n" | tee -a "$MONITOR_LOG"

    LAST_CHECK_TIME=$CURRENT_TIME
done
