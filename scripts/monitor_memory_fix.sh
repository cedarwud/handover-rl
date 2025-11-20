#!/bin/bash
# Monitor memory fix test progress

LOG_FILE="/home/sat/satellite/handover-rl/logs/memory_fix_test.log"

echo "========================================="
echo "Memory Fix Test Progress Monitor"
echo "========================================="
echo ""

# Extract current episode from log
CURRENT_EP=$(grep -oP 'Training:\s+\d+%\|.*?\|\s+\K\d+(?=/1000)' "$LOG_FILE" 2>/dev/null | tail -1)
if [ -z "$CURRENT_EP" ]; then
    CURRENT_EP=0
fi

echo "Current Episode: $CURRENT_EP / 1000"
echo "Progress: $(echo "scale=1; $CURRENT_EP * 100 / 1000" | bc)%"
echo ""

# Check for resource errors
RESOURCE_ERRORS=$(grep -c "RESOURCE ERROR" "$LOG_FILE" 2>/dev/null || echo 0)
TIMEOUT_ERRORS=$(grep -c "TIMEOUT" "$LOG_FILE" 2>/dev/null || echo 0)
TOTAL_SKIPPED=$(grep -c "skipped due to" "$LOG_FILE" 2>/dev/null || echo 0)

echo "Skipped Episodes:"
echo "  Memory errors: $RESOURCE_ERRORS"
echo "  Timeout errors: $TIMEOUT_ERRORS"
echo "  Total skipped: $TOTAL_SKIPPED"
echo ""

# Check Episode 522 specifically
if grep -q "Episode 522" "$LOG_FILE" 2>/dev/null; then
    echo "📍 Episode 522 Status:"
    if grep -q "Episode 522.*TIMEOUT" "$LOG_FILE" 2>/dev/null; then
        echo "   ⏱️  TIMEOUT - Auto-skipped"
    elif grep -q "Episode 522.*RESOURCE" "$LOG_FILE" 2>/dev/null; then
        echo "   💥 RESOURCE ERROR - Auto-skipped"
    elif grep -q "Episode 522.*skipped" "$LOG_FILE" 2>/dev/null; then
        echo "   ⚠️  SKIPPED (other reason)"
    else
        echo "   ✅ Passed Episode 522 successfully!"
    fi
    echo ""
fi

# Show memory usage trend (if resource errors exist)
if [ "$RESOURCE_ERRORS" -gt 0 ]; then
    echo "Memory Usage When Errors Started:"
    grep "RESOURCE ERROR: Memory usage" "$LOG_FILE" 2>/dev/null | head -5 | awk -F'Memory usage ' '{print $2}' | awk '{print "  "$1}'
    echo ""
fi

echo "========================================="
echo "Live log: tail -f $LOG_FILE"
echo "========================================="
