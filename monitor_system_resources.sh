#!/bin/bash
# Monitor system resources during training
# Helps identify I/O bottlenecks, memory issues, or GPU problems

LOG_FILE="/tmp/system_monitor.log"

echo "=========================================="
echo "System Resource Monitor"
echo "=========================================="
echo "Logging to: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Clear previous log
> $LOG_FILE

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Count training processes
    TRAIN_PROCS=$(pgrep -f "train_sb3.py.*academic" | wc -l)

    # Memory usage
    MEM_INFO=$(free -h | grep "Mem:" | awk '{print $3 "/" $2}')
    SWAP_INFO=$(free -h | grep "Swap:" | awk '{print $3 "/" $2}')

    # GPU usage
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | sed 's/, /\//' || echo "N/A")

    # Disk I/O (if iostat available)
    if command -v iostat &> /dev/null; then
        DISK_UTIL=$(iostat -x 1 2 | grep -A 1 "Device" | tail -1 | awk '{print $14}')
    else
        DISK_UTIL="N/A"
    fi

    # Log to file
    echo "$TIMESTAMP | Procs: $TRAIN_PROCS | Mem: $MEM_INFO | Swap: $SWAP_INFO | GPU: ${GPU_UTIL}% (${GPU_MEM}MB) | Disk: ${DISK_UTIL}%" >> $LOG_FILE

    # Display on screen
    clear
    echo "=========================================="
    echo "System Resource Monitor - $TIMESTAMP"
    echo "=========================================="
    echo ""
    echo "Training Processes: $TRAIN_PROCS"
    echo ""
    echo "Memory:"
    echo "  RAM:  $MEM_INFO"
    echo "  Swap: $SWAP_INFO"
    echo ""
    echo "GPU:"
    echo "  Utilization: ${GPU_UTIL}%"
    echo "  Memory:      ${GPU_MEM} MB"
    echo ""
    echo "Disk Utilization: ${DISK_UTIL}%"
    echo ""
    echo "=========================================="
    echo "Recent FPS (last 5 samples per seed):"
    echo "=========================================="

    for seed in 42 123 456 789 2024; do
        if [ -f "/tmp/academic_seed$seed.log" ]; then
            FPS=$(grep "| *fps" /tmp/academic_seed$seed.log 2>/dev/null | tail -5 | awk '{print $3}' | sed 's/|//g' | tr '\n' ', ')
            TIMESTEPS=$(grep "total_timesteps" /tmp/academic_seed$seed.log 2>/dev/null | tail -1 | awk '{print $3}' | sed 's/|//g')
            echo "  Seed $seed: FPS=[$FPS] Timesteps=$TIMESTEPS"
        fi
    done

    echo ""
    echo "=========================================="
    echo "Log file: $LOG_FILE"
    echo "Press Ctrl+C to stop"
    echo "=========================================="

    sleep 10
done
