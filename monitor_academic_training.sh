#!/bin/bash
# Monitor Academic Standard Training Progress

echo "=========================================="
echo "Academic Training Progress Monitor"
echo "=========================================="
echo ""

SEEDS=(42 123 456 789 2024)

while true; do
    clear
    echo "=========================================="
    echo "Academic Training Progress"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""

    # Check if training processes are running
    RUNNING=0
    for seed in "${SEEDS[@]}"; do
        if pgrep -f "train_sb3.py.*academic_seed${seed}" > /dev/null; then
            RUNNING=$((RUNNING + 1))
        fi
    done

    echo "Running processes: $RUNNING / 5"
    echo ""

    # Show progress for each seed
    for seed in "${SEEDS[@]}"; do
        LOG="/tmp/academic_seed${seed}.log"

        if [ -f "$LOG" ]; then
            echo "Seed $seed:"

            # Check if process is running
            if pgrep -f "train_sb3.py.*academic_seed${seed}" > /dev/null; then
                echo "  Status: RUNNING"

                # Try to extract progress from log
                PROGRESS=$(grep -oE "[0-9]+%" "$LOG" | tail -1 || echo "Starting...")
                echo "  Progress: $PROGRESS"

                # Check if training has started (look for DQN model creation)
                if grep -q "Using cuda device" "$LOG"; then
                    echo "  Stage: Training in progress"

                    # Try to find episode info
                    EPISODES=$(grep -oE "Episode [0-9]+" "$LOG" | tail -1 || echo "N/A")
                    if [ "$EPISODES" != "N/A" ]; then
                        echo "  $EPISODES"
                    fi
                else
                    echo "  Stage: Initializing..."
                fi
            else
                # Check if completed
                if grep -q "✅ Training completed" "$LOG" 2>/dev/null; then
                    echo "  Status: COMPLETED"
                    MODEL="output/academic_seed${seed}/models/dqn_final.zip"
                    if [ -f "$MODEL" ]; then
                        echo "  Model saved: $MODEL"
                    fi
                else
                    echo "  Status: STOPPED (check log for errors)"
                fi
            fi
        else
            echo "Seed $seed: Log not found"
        fi
        echo ""
    done

    echo "=========================================="
    echo "Monitor Commands:"
    echo "  tail -f /tmp/academic_seed42.log   # View seed 42 log"
    echo "  tail -f /tmp/academic_seed123.log  # View seed 123 log"
    echo "  Press Ctrl+C to stop monitoring"
    echo "=========================================="

    # Exit if all training completed
    if [ $RUNNING -eq 0 ]; then
        echo ""
        echo "All training processes have finished!"
        break
    fi

    sleep 30
done
