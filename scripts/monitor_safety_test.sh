#!/bin/bash
# Monitor safety test progress

LOG_FILE="/home/sat/satellite/handover-rl/logs/safety_test_full.log"

echo "========================================="
echo "Safety Test Progress Monitor"
echo "========================================="
echo ""

# Extract current episode from log
CURRENT_EP=$(grep -oP 'Episode \K\d+(?=:)' "$LOG_FILE" 2>/dev/null | tail -1)
if [ -z "$CURRENT_EP" ]; then
    CURRENT_EP=0
fi

echo "Current Episode: $CURRENT_EP / 1000"
echo "Progress: $(echo "scale=1; $CURRENT_EP * 100 / 1000" | bc)%"
echo ""

# Check for Episode 522
if grep -q "Episode 522" "$LOG_FILE" 2>/dev/null; then
    echo "📍 Episode 522 Status:"
    if grep -q "Episode 522.*TIMEOUT" "$LOG_FILE" 2>/dev/null; then
        echo "   ⏱️  TIMEOUT - Auto-skipped after 600s"
    elif grep -q "Episode 522.*RESOURCE" "$LOG_FILE" 2>/dev/null; then
        echo "   💥 RESOURCE ERROR - Auto-skipped"
    elif grep -q "Episode 522.*CRASHED" "$LOG_FILE" 2>/dev/null; then
        echo "   ❌ CRASHED - Auto-skipped"
    elif grep -q "Episode 522.*skipped" "$LOG_FILE" 2>/dev/null; then
        echo "   ⚠️  SKIPPED due to error"
    else
        echo "   ✅ In progress or completed successfully"
    fi
    echo ""
fi

# Count skipped episodes
SKIPPED=$(grep -c "skipped due to" "$LOG_FILE" 2>/dev/null || echo 0)
echo "Skipped Episodes: $SKIPPED"
echo ""

# Show recent episodes
echo "Recent Episodes (last 10 lines):"
grep "Episode [0-9]" "$LOG_FILE" 2>/dev/null | tail -10
echo ""

# Estimated time remaining
if [ "$CURRENT_EP" -gt 0 ]; then
    ELAPSED=$(grep "Starting training" "$LOG_FILE" -A 1000 | head -1 | awk '{print $1, $2}')
    echo "Training started: $ELAPSED"

    # Rough estimate: 13s per episode
    REMAINING=$((1000 - CURRENT_EP))
    REMAINING_SECONDS=$((REMAINING * 13))
    REMAINING_HOURS=$(echo "scale=1; $REMAINING_SECONDS / 3600" | bc)

    echo "Estimated time remaining: ~${REMAINING_HOURS} hours"
fi

echo ""
echo "========================================="
echo "To watch live: tail -f $LOG_FILE"
echo "========================================="
