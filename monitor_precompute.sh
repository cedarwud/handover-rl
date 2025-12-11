#!/bin/bash
# Monitor Precompute Generation Progress

echo "=========================================="
echo "Precompute Generation Monitor"
echo "=========================================="
echo ""

while true; do
    clear
    echo "=========================================="
    echo "Precompute Generation Progress"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""

    # Check if process is running
    if pgrep -f "generate_orbit_precompute.py" > /dev/null; then
        echo "Status: RUNNING"
        echo ""

        # Show latest progress from log
        echo "Latest Progress:"
        grep -E "(\%|sat/s|Satellites:|✅|ERROR)" /tmp/precompute_generation.log | tail -5
        echo ""

        # Check output file size
        if [ -f "data/orbit_precompute_30days.h5" ]; then
            SIZE=$(du -h data/orbit_precompute_30days.h5 | cut -f1)
            echo "Output file size: $SIZE (target: ~2.5GB)"
        fi
    else
        echo "Status: STOPPED"
        echo ""

        # Check if completed successfully
        if grep -q "✅.*completed" /tmp/precompute_generation.log 2>/dev/null; then
            echo "Generation COMPLETED!"
            if [ -f "data/orbit_precompute_30days.h5" ]; then
                SIZE=$(du -h data/orbit_precompute_30days.h5 | cut -f1)
                echo "Final file size: $SIZE"
            fi
            break
        elif grep -q "ERROR" /tmp/precompute_generation.log 2>/dev/null; then
            echo "Generation FAILED - check log for errors"
            break
        else
            echo "Generation STOPPED unexpectedly"
            break
        fi
    fi

    echo ""
    echo "=========================================="
    echo "Press Ctrl+C to stop monitoring"
    echo "View full log: tail -f /tmp/precompute_generation.log"
    echo "=========================================="

    sleep 30
done
