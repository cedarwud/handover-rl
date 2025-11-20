#!/bin/bash
# Monitor Episode 524 Performance
# Purpose: Track if HDF5 optimization fixes the I/O bottleneck

LOG_FILE="output/level4_optimized_20251115/training.log"
PERF_LOG="/tmp/episode524_performance.log"

echo "======================================" | tee "$PERF_LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Episode 524 Performance Monitor Started" | tee -a "$PERF_LOG"
echo "======================================" | tee -a "$PERF_LOG"
echo "" | tee -a "$PERF_LOG"

# Wait until Episode 520
while true; do
    sleep 60

    # Get current episode
    current_ep=$(tail -50 "$LOG_FILE" 2>/dev/null | grep -oP 'Training:\s+\d+%.*?\|\s+\K\d+(?=/1000)' | tail -1)

    if [ -z "$current_ep" ]; then
        continue
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Current episode: $current_ep" | tee -a "$PERF_LOG"

    # Start detailed monitoring when approaching Episode 520
    if [ "$current_ep" -ge 520 ] && [ "$current_ep" -le 530 ]; then
        echo "" | tee -a "$PERF_LOG"
        echo "========== Episode $current_ep Performance ==========" | tee -a "$PERF_LOG"

        # Extract timing info from progress bar
        timing_info=$(tail -10 "$LOG_FILE" 2>/dev/null | grep "Training:" | tail -1)
        echo "$timing_info" | tee -a "$PERF_LOG"

        # Check system load
        echo "System load: $(uptime | awk -F'load average:' '{print $2}')" | tee -a "$PERF_LOG"

        # Check process status
        pid=$(pgrep -f "python train.py.*level 4" | head -1)
        if [ -n "$pid" ]; then
            echo "Process CPU: $(ps -p $pid -o %cpu= | xargs)%" | tee -a "$PERF_LOG"
            echo "Process MEM: $(ps -p $pid -o %mem= | xargs)%" | tee -a "$PERF_LOG"
        fi

        echo "===================================================" | tee -a "$PERF_LOG"
        echo "" | tee -a "$PERF_LOG"

        # Special handling for Episode 524
        if [ "$current_ep" -eq 524 ]; then
            echo "🎯 Episode 524 detected! Monitoring intensively..." | tee -a "$PERF_LOG"

            # Monitor for 5 minutes
            for i in {1..5}; do
                sleep 60
                new_ep=$(tail -50 "$LOG_FILE" 2>/dev/null | grep -oP 'Training:\s+\d+%.*?\|\s+\K\d+(?=/1000)' | tail -1)

                if [ "$new_ep" -gt 524 ]; then
                    echo "✅ Episode 524 completed successfully!" | tee -a "$PERF_LOG"
                    timing=$(tail -10 "$LOG_FILE" 2>/dev/null | grep "525/1000" | grep -oP '\d+\.\d+s/it')
                    echo "   Episode 524 timing: $timing" | tee -a "$PERF_LOG"
                    break
                else
                    echo "   [T+${i}min] Still on episode $new_ep..." | tee -a "$PERF_LOG"
                fi
            done

            final_ep=$(tail -50 "$LOG_FILE" 2>/dev/null | grep -oP 'Training:\s+\d+%.*?\|\s+\K\d+(?=/1000)' | tail -1)
            if [ "$final_ep" -eq 524 ]; then
                echo "⚠️  Episode 524 STUCK after 5 minutes!" | tee -a "$PERF_LOG"
            fi
        fi
    fi

    # Stop monitoring after Episode 530
    if [ "$current_ep" -gt 530 ]; then
        echo "" | tee -a "$PERF_LOG"
        echo "======================================" | tee -a "$PERF_LOG"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Monitoring complete (passed Episode 530)" | tee -a "$PERF_LOG"
        echo "======================================" | tee -a "$PERF_LOG"
        break
    fi
done
