#!/bin/bash
# Monitor Training Progress
# Usage: ./tools/monitor_training.sh [log_file]

LOG_FILE="${1:-/tmp/level2_training.log}"

echo "======================================"
echo "Training Progress Monitor"
echo "======================================"
echo "Log file: $LOG_FILE"
echo "Time: $(date)"
echo ""

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    exit 1
fi

# Extract latest progress
echo "📊 Latest Progress:"
tail -100 "$LOG_FILE" | grep -E "Training:|Episode [0-9]+/|Loss:|Reward:" | tail -10

echo ""
echo "📈 Episode Summary:"
grep -E "Episode [0-9]+/" "$LOG_FILE" | tail -5

echo ""
echo "🎯 Latest Metrics:"
tail -50 "$LOG_FILE" | grep -E "avg_reward|avg_loss|epsilon" | tail -3

echo ""
echo "⏱️  Estimated Progress:"
EPISODES_DONE=$(grep -c "Episode [0-9]\+/" "$LOG_FILE")
echo "Episodes completed: $EPISODES_DONE"

echo ""
echo "======================================"
echo "Use: watch -n 30 ./tools/monitor_training.sh"
echo "======================================"
