#!/bin/bash
# Diagnostic script to capture detailed info when Episode 524 is reached
# This will help identify the root cause of the stuck bug

LOG_FILE="/tmp/level4_training_monitored.log"
DIAG_LOG="/tmp/episode524_diagnostic.log"
MONITOR_INTERVAL=30  # Check every 30 seconds when approaching Episode 524

echo "======================================" | tee -a "$DIAG_LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Episode 524 Diagnostic Monitor Started" | tee -a "$DIAG_LOG"
echo "======================================" | tee -a "$DIAG_LOG"

# Get PID of training process
get_train_pid() {
    pgrep -f "python train.py.*level 4" | head -1
}

# Get current episode number
get_current_episode() {
    tail -20 "$LOG_FILE" 2>/dev/null | grep -oP 'Training:\s+\d+%.*?\|\s+\K\d+(?=/1000)' | tail -1
}

# Capture system state
capture_system_state() {
    local episode=$1
    local pid=$2

    echo "" | tee -a "$DIAG_LOG"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==== Episode $episode System State ====" | tee -a "$DIAG_LOG"

    # Memory usage
    echo "Memory (python process):" | tee -a "$DIAG_LOG"
    ps -p $pid -o pid,rss,vsz,%mem,cmd | tee -a "$DIAG_LOG"

    # CPU usage
    echo "" | tee -a "$DIAG_LOG"
    echo "CPU (python process):" | tee -a "$DIAG_LOG"
    ps -p $pid -o pid,%cpu,time,cmd | tee -a "$DIAG_LOG"

    # System load
    echo "" | tee -a "$DIAG_LOG"
    echo "System Load:" | tee -a "$DIAG_LOG"
    uptime | tee -a "$DIAG_LOG"

    # Disk I/O (if iostat available)
    if command -v iostat &> /dev/null; then
        echo "" | tee -a "$DIAG_LOG"
        echo "Disk I/O:" | tee -a "$DIAG_LOG"
        iostat -x 1 2 | tail -10 | tee -a "$DIAG_LOG"
    fi

    # GPU usage
    if command -v nvidia-smi &> /dev/null; then
        echo "" | tee -a "$DIAG_LOG"
        echo "GPU Usage:" | tee -a "$DIAG_LOG"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | tee -a "$DIAG_LOG"
    fi

    # Last 5 log lines
    echo "" | tee -a "$DIAG_LOG"
    echo "Last log lines:" | tee -a "$DIAG_LOG"
    tail -5 "$LOG_FILE" | tee -a "$DIAG_LOG"

    echo "======================================" | tee -a "$DIAG_LOG"
}

# Main monitoring loop
last_episode=0
while true; do
    sleep $MONITOR_INTERVAL

    current_episode=$(get_current_episode)

    if [ -z "$current_episode" ]; then
        continue
    fi

    # Start detailed monitoring when approaching Episode 524
    if [ "$current_episode" -ge 510 ] && [ "$current_episode" -le 530 ]; then
        pid=$(get_train_pid)

        if [ -n "$pid" ]; then
            # Capture state for episodes 510, 512, 515, 520, 522, 523, 524, 525, 526, 530
            if [ "$current_episode" -ne "$last_episode" ]; then
                case $current_episode in
                    510|512|515|520|522|523|524|525|526|530)
                        capture_system_state $current_episode $pid
                        ;;
                esac
                last_episode=$current_episode
            fi

            # Extra frequent monitoring for Episode 524
            if [ "$current_episode" -eq 524 ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  Episode 524 detected! Continuous monitoring..." | tee -a "$DIAG_LOG"

                # Monitor every 10 seconds during Episode 524
                for i in {1..12}; do  # 12 iterations = 2 minutes
                    sleep 10
                    capture_system_state "524 (T+${i}0s)" $pid

                    # Check if still on Episode 524
                    new_ep=$(get_current_episode)
                    if [ "$new_ep" -ne 524 ]; then
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Episode 524 completed successfully!" | tee -a "$DIAG_LOG"
                        break
                    fi
                done

                # If still on 524 after 2 minutes, something is wrong
                final_ep=$(get_current_episode)
                if [ "$final_ep" -eq 524 ]; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚨 Episode 524 STUCK for >2 minutes!" | tee -a "$DIAG_LOG"
                fi
            fi
        fi
    fi

    # Exit monitoring after Episode 530
    if [ "$current_episode" -gt 530 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Monitoring complete (passed Episode 530)" | tee -a "$DIAG_LOG"
        break
    fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Diagnostic monitor exiting" | tee -a "$DIAG_LOG"
