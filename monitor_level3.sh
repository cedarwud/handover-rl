#!/bin/bash
echo "========================================"
echo "Level 3 Training Monitor"
echo "Started: $(date)"
echo "========================================"
echo ""

while true; do
    if [ -f "training_level3_stable.log" ]; then
        # Get latest episode
        latest=$(grep "Episode.*reward" training_level3_stable.log | tail -1)
        
        if [ ! -z "$latest" ]; then
            echo "[$(date +%H:%M:%S)] $latest"
            
            # Check if completed
            if echo "$latest" | grep -q "500/500"; then
                echo ""
                echo "✅ Training completed!"
                echo ""
                echo "=== Final Statistics ==="
                grep "Episode.*reward" training_level3_stable.log | tail -5
                echo ""
                echo "=== Best Model Info ==="
                grep -E "best model|Best reward" training_level3_stable.log | tail -3
                break
            fi
        fi
    fi
    
    sleep 60  # Check every minute
done
