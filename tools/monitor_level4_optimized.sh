#!/bin/bash
# Monitor Level 4 training with optimized HDF5
# Focus: Episodes 522-532 (previously problematic)

LOG_FILE="output/level4_optimized_final.log"

echo "=========================================="
echo "Level 4 Training Monitor (Optimized HDF5)"
echo "=========================================="
echo ""

# Training PID
PID=$(ps aux | grep "train.py" | grep "level4_optimized_final" | grep -v grep | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "❌ Training process not found"
    exit 1
fi
echo "✅ Training running (PID: $PID)"
echo ""

# Current episode
CURRENT_EP=$(grep "Training:" "$LOG_FILE" | tail -1 | grep -oP '\d+(?=/1000)')
if [ -z "$CURRENT_EP" ]; then
    CURRENT_EP=0
fi
echo "📊 Current Episode: $CURRENT_EP / 1000"
echo ""

# Time per episode (last 5 episodes)
echo "⏱️  Recent Episode Times:"
grep "Training:" "$LOG_FILE" | tail -5 | grep -oP '\d+\.\d+(?=s/it)' | awk '{sum+=$1; count++} END {if(count>0) printf "   Last 5 avg: %.2fs/episode\n", sum/count}'
echo ""

# Episodes 522-532 status
echo "🎯 Critical Range (Episodes 522-532):"
if [ "$CURRENT_EP" -lt 522 ]; then
    REMAINING=$((522 - CURRENT_EP))
    echo "   Status: Not yet reached"
    echo "   Episodes until 522: $REMAINING"
elif [ "$CURRENT_EP" -ge 522 ] && [ "$CURRENT_EP" -le 532 ]; then
    echo "   Status: 🚨 IN CRITICAL RANGE"
    echo "   Current: Episode $CURRENT_EP"
    echo "   Progress: $((CURRENT_EP - 521))/11"
elif [ "$CURRENT_EP" -gt 532 ]; then
    echo "   Status: ✅ PASSED (Episode $CURRENT_EP)"
fi
echo ""

# Episode 522 details (if reached)
if [ "$CURRENT_EP" -ge 522 ]; then
    echo "📈 Episodes 522-532 Performance:"
    for ep in {522..532}; do
        TIME=$(grep -A1 "Episode start time.*2025-10-13" "$LOG_FILE" | grep "Training:" | grep -oP "${ep}/1000.*?\K\d+\.\d+(?=s/it)" | head -1)
        if [ ! -z "$TIME" ]; then
            echo "   Episode $ep: ${TIME}s"
        fi
    done
    echo ""
fi

# Estimated completion
if [ ! -z "$CURRENT_EP" ]; then
    REMAINING=$((1000 - CURRENT_EP))
    AVG_TIME=$(grep "Training:" "$LOG_FILE" | tail -10 | grep -oP '\d+\.\d+(?=s/it)' | awk '{sum+=$1; count++} END {if(count>0) print sum/count}')
    if [ ! -z "$AVG_TIME" ]; then
        REMAINING_HOURS=$(echo "$REMAINING * $AVG_TIME / 3600" | bc -l)
        printf "⏳ Estimated time remaining: %.1f hours\n" $REMAINING_HOURS
    fi
fi
echo ""

# Latest log lines
echo "📝 Latest Log:"
tail -3 "$LOG_FILE"
echo ""
echo "=========================================="
