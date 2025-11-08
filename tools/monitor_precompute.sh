#!/bin/bash
# Monitor Precompute Generation Progress
# Usage: ./tools/monitor_precompute.sh [interval_seconds]

INTERVAL=${1:-5}  # Default: 5 seconds

echo "=========================================="
echo "Precompute Generation Monitor"
echo "=========================================="
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "Precompute Generation Progress"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""

    # Check for running precompute processes
    RUNNING=$(ps aux | grep "generate_orbit_precompute.py" | grep -v grep)

    if [ -z "$RUNNING" ]; then
        echo "⚠️  No precompute generation process running"
        echo ""

        # Check for completed files
        if [ -f "data/orbit_precompute_1day_test.h5" ]; then
            SIZE=$(du -h data/orbit_precompute_1day_test.h5 | cut -f1)
            echo "✅ 1-day test table completed: $SIZE"
        fi

        if [ -f "data/orbit_precompute_7days.h5" ]; then
            SIZE=$(du -h data/orbit_precompute_7days.h5 | cut -f1)
            echo "✅ 7-day table completed: $SIZE"
        fi

        echo ""
        echo "To start generation:"
        echo "  python scripts/generate_orbit_precompute.py \\"
        echo "    --start-time \"2025-10-07 00:00:00\" \\"
        echo "    --end-time \"2025-10-14 00:00:00\" \\"
        echo "    --output data/orbit_precompute_7days.h5 \\"
        echo "    --config config/diagnostic_config.yaml \\"
        echo "    --yes"

        break
    else
        echo "🔄 Active Process:"
        echo "$RUNNING" | awk '{print "   PID: " $2 "  CPU: " $3 "%  MEM: " $4 "%"}'
        echo ""

        # Check output files being generated
        echo "📁 Output Files:"
        for f in data/orbit_precompute_*.h5; do
            if [ -f "$f" ]; then
                SIZE=$(du -h "$f" 2>/dev/null | cut -f1)
                MTIME=$(stat -c %y "$f" 2>/dev/null | cut -d. -f1)
                echo "   $(basename $f): $SIZE (updated: $MTIME)"
            fi
        done
        echo ""

        # Try to extract progress from log (last few lines)
        echo "📊 Latest Progress:"
        # Look for progress in running process output
        tail -20 /tmp/precompute_*.log 2>/dev/null | grep "Satellites:" | tail -1 || echo "   (checking...)"

        # Alternative: check process output if logged to file
        if [ -f "precompute_generation.log" ]; then
            grep "Satellites:" precompute_generation.log | tail -1
        fi

        echo ""
        echo "=========================================="
        echo "Next update in $INTERVAL seconds..."
        echo "=========================================="
    fi

    sleep $INTERVAL
done
