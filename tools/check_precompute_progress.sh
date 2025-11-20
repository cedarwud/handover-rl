#!/bin/bash
# Quick check of precompute generation progress
# Usage: ./tools/check_precompute_progress.sh

echo "=========================================="
echo "Precompute Generation Progress"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# Check if process is running
RUNNING=$(ps aux | grep "generate_orbit_precompute.py" | grep -v grep)

if [ -z "$RUNNING" ]; then
    echo "❌ No precompute generation process running"
    echo ""

    # Check for completed files
    if [ -f "data/orbit_precompute_7days.h5" ]; then
        SIZE=$(du -h data/orbit_precompute_7days.h5 | cut -f1)
        echo "✅ 7-day table: $SIZE"

        # Check if it's the expected size (~537 MB)
        SIZE_BYTES=$(du -b data/orbit_precompute_7days.h5 | cut -f1)
        EXPECTED_BYTES=563000000  # ~537 MB

        if [ $SIZE_BYTES -ge $EXPECTED_BYTES ]; then
            echo "✅ Generation COMPLETE (expected size reached)"
        else
            echo "⚠️  File exists but may be incomplete"
            echo "   Current: $SIZE, Expected: ~537 MB"
        fi
    else
        echo "❌ 7-day table not found"
    fi

    if [ -f "data/orbit_precompute_1day_test.h5" ]; then
        SIZE=$(du -h data/orbit_precompute_1day_test.h5 | cut -f1)
        echo "✅ 1-day test table: $SIZE"
    fi
else
    echo "🔄 Generation in progress..."
    echo ""

    # Show process info
    echo "Process:"
    echo "$RUNNING" | awk '{print "   PID: " $2 "  CPU: " $3 "%  MEM: " $4 "%"}'
    echo ""

    # Check current file size
    if [ -f "data/orbit_precompute_7days.h5" ]; then
        SIZE=$(du -h data/orbit_precompute_7days.h5 | cut -f1)
        SIZE_BYTES=$(du -b data/orbit_precompute_7days.h5 | cut -f1)
        EXPECTED_BYTES=563000000  # ~537 MB

        PERCENT=$((SIZE_BYTES * 100 / EXPECTED_BYTES))

        echo "📊 Progress:"
        echo "   File size: $SIZE / ~537 MB"
        echo "   Progress: ~$PERCENT%"
        echo ""

        # Estimate time remaining (rough)
        if [ $SIZE_BYTES -gt 10000000 ]; then  # > 10 MB
            echo "⏱️  Estimated time: ~42-49 minutes total"
            echo "   (using 31 parallel processes)"
        fi
    else
        echo "📊 Initializing..."
        echo "   (HDF5 file will appear soon)"
    fi
fi

echo ""
echo "=========================================="
