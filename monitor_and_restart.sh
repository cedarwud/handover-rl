#!/bin/bash
# Training Monitor and Auto-Restart Script
# 監控訓練進度，檢測卡住情況並自動重啟

CONFIG_FILE="config/diagnostic_config.yaml"
LEVEL=4
OUTPUT_DIR="output/level4_20251112"
LOG_FILE="/tmp/level4_training_monitored.log"
MONITOR_LOG="/tmp/training_monitor.log"
CHECK_INTERVAL=60  # 每 60 秒檢查一次
STUCK_THRESHOLD=300  # 如果 5 分鐘沒有新進度，視為卡住

# 獲取最新的 checkpoint
get_latest_checkpoint() {
    local checkpoint_dir="$OUTPUT_DIR/checkpoints"
    if [ -d "$checkpoint_dir" ]; then
        local latest=$(ls -t "$checkpoint_dir"/checkpoint_ep*.pth 2>/dev/null | head -1)
        echo "$latest"
    fi
}

# 檢查訓練是否在運行
is_training_running() {
    pgrep -f "python train.py.*level 4" > /dev/null
    return $?
}

# 獲取當前 episode 數
get_current_episode() {
    tail -20 "$LOG_FILE" 2>/dev/null | grep -oP 'Training:\s+\d+%.*?\|\s+\K\d+(?=/1000)' | tail -1
}

# 獲取日誌最後修改時間
get_log_age() {
    if [ -f "$LOG_FILE" ]; then
        echo $(($(date +%s) - $(stat -c %Y "$LOG_FILE")))
    else
        echo 0
    fi
}

# 啟動訓練
start_training() {
    local checkpoint="$1"
    local resume_flag=""

    if [ -n "$checkpoint" ] && [ -f "$checkpoint" ]; then
        resume_flag="--resume $checkpoint"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 從 checkpoint 繼續訓練: $checkpoint" | tee -a "$MONITOR_LOG"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 開始新的訓練" | tee -a "$MONITOR_LOG"
    fi

    cd /home/sat/satellite/handover-rl
    source venv/bin/activate

    nice -n 10 python train.py \
        --algorithm dqn \
        --level $LEVEL \
        --config "$CONFIG_FILE" \
        --output-dir "$OUTPUT_DIR" \
        $resume_flag \
        > "$LOG_FILE" 2>&1 &

    local pid=$!
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 訓練已啟動，PID: $pid" | tee -a "$MONITOR_LOG"
}

# 停止訓練
stop_training() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 停止訓練程序..." | tee -a "$MONITOR_LOG"
    pkill -f "python train.py.*level 4"
    sleep 5

    # 強制 kill 如果還在運行
    if is_training_running; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 強制停止..." | tee -a "$MONITOR_LOG"
        pkill -9 -f "python train.py.*level 4"
        sleep 2
    fi
}

# 主監控循環
echo "========================================" | tee -a "$MONITOR_LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 訓練監控啟動" | tee -a "$MONITOR_LOG"
echo "  檢查間隔: ${CHECK_INTERVAL}s" | tee -a "$MONITOR_LOG"
echo "  卡住閾值: ${STUCK_THRESHOLD}s" | tee -a "$MONITOR_LOG"
echo "========================================" | tee -a "$MONITOR_LOG"

# 首次啟動
latest_checkpoint=$(get_latest_checkpoint)
start_training "$latest_checkpoint"

last_episode=0
stuck_count=0

# 監控循環
while true; do
    sleep $CHECK_INTERVAL

    # 檢查是否還在運行
    if ! is_training_running; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  訓練程序已停止" | tee -a "$MONITOR_LOG"

        # 檢查是否正常完成
        if grep -q "Training: 100%" "$LOG_FILE" 2>/dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 訓練已完成！" | tee -a "$MONITOR_LOG"
            break
        fi

        # 異常停止，重啟
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔄 準備重啟..." | tee -a "$MONITOR_LOG"
        sleep 10
        latest_checkpoint=$(get_latest_checkpoint)
        start_training "$latest_checkpoint"
        stuck_count=0
        continue
    fi

    # 檢查是否卡住
    current_episode=$(get_current_episode)
    log_age=$(get_log_age)

    if [ -n "$current_episode" ]; then
        if [ "$current_episode" -eq "$last_episode" ] && [ "$log_age" -gt "$STUCK_THRESHOLD" ]; then
            stuck_count=$((stuck_count + 1))
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  訓練可能卡住了 (Episode $current_episode, 已等待 ${log_age}s, 次數: $stuck_count)" | tee -a "$MONITOR_LOG"

            # 連續 2 次檢測到卡住才重啟（避免誤判）
            if [ $stuck_count -ge 2 ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔄 訓練卡住，準備重啟..." | tee -a "$MONITOR_LOG"
                stop_training
                sleep 10
                latest_checkpoint=$(get_latest_checkpoint)
                start_training "$latest_checkpoint"
                stuck_count=0
            fi
        else
            if [ "$current_episode" -ne "$last_episode" ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 訓練正常進行中 (Episode $current_episode)" | tee -a "$MONITOR_LOG"
                stuck_count=0
            fi
            last_episode=$current_episode
        fi
    fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 監控程序結束" | tee -a "$MONITOR_LOG"
