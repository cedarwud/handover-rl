#!/bin/bash
# Level 4 Training Monitor (1000 episodes)
echo "========================================"
echo "Level 4 Official Baseline Training"
echo "Started: $(date)"
echo "Target: 1000 episodes"
echo "Estimated time: ~5 hours"
echo "========================================"
echo ""

while true; do
    if [ -f "training_level4_official.log" ]; then
        # Get latest episode
        latest=$(grep "Episode.*reward" training_level4_official.log | tail -1)

        if [ ! -z "$latest" ]; then
            echo "[$(date +%H:%M:%S)] $latest"

            # Check if completed
            if echo "$latest" | grep -q "1000/1000"; then
                echo ""
                echo "✅ Level 4 Training Completed!"
                echo ""
                echo "=== Final Statistics ==="
                grep "Episode.*reward" training_level4_official.log | tail -5
                echo ""
                echo "=== Best Model Info ==="
                grep -E "best model|Best reward" training_level4_official.log | tail -3
                echo ""
                echo "Next step: Evaluate with fixed evaluation script"
                break
            fi
        fi
    fi

    sleep 180  # Check every 3 minutes
done
