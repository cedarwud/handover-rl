#!/bin/bash
# Monitor batch training progress

LOG_FILE="${1:-/home/sat/satellite/handover-rl/logs/batch_test.log}"

echo "========================================="
echo "Batch Training Progress Monitor"
echo "========================================="
echo ""

# Current batch
CURRENT_BATCH=$(grep "Starting Batch" "$LOG_FILE" 2>/dev/null | tail -1 | awk '{print $3}')
if [ -z "$CURRENT_BATCH" ]; then
    CURRENT_BATCH="0/?"
fi

echo "Current Batch: $CURRENT_BATCH"
echo ""

# Batch completion
COMPLETED_BATCHES=$(grep -c "Batch.*completed successfully" "$LOG_FILE" 2>/dev/null || echo 0)
echo "Completed Batches: $COMPLETED_BATCHES"
echo ""

# Current episode in current batch
CURRENT_EP=$(grep "Training:.*%" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '\d+/\d+' | head -1)
if [ -n "$CURRENT_EP" ]; then
    echo "Current Batch Progress: $CURRENT_EP episodes"
    echo ""
fi

# Show recent batch completions
echo "Recent Activity:"
grep -E "(Starting Batch|completed successfully)" "$LOG_FILE" 2>/dev/null | tail -5
echo ""

echo "========================================="
echo "Live log: tail -f $LOG_FILE"
echo "========================================="
