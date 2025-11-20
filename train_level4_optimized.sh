#!/bin/bash
# Level 4 Training with HDF5 Optimization
# Fixed: Episode 524 I/O bottleneck by increasing HDF5 chunk cache to 512 MB

CONFIG_FILE="config/diagnostic_config.yaml"
LEVEL=4
OUTPUT_DIR="output/level4_optimized_$(date +%Y%m%d)"
LOG_FILE="${OUTPUT_DIR}/training.log"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo "========================================"
echo "Level 4 Training (HDF5 Optimized)"
echo "========================================"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

# Find latest checkpoint if exists
LATEST_CHECKPOINT=""
if [ -d "$OUTPUT_DIR/checkpoints" ]; then
    LATEST_CHECKPOINT=$(ls -t "$OUTPUT_DIR/checkpoints"/checkpoint_ep*.pth 2>/dev/null | head -1)
fi

# Build command
CMD="python train.py --algorithm dqn --level $LEVEL --config $CONFIG_FILE --output-dir $OUTPUT_DIR"

if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "Found checkpoint: $LATEST_CHECKPOINT"
    echo "Resuming from checkpoint..."
    CMD="$CMD --resume $LATEST_CHECKPOINT"
else
    echo "No checkpoint found, starting fresh training..."
fi

echo ""
echo "Starting training..."
echo "Command: $CMD"
echo ""

# Activate virtual environment
source venv/bin/activate

# Run training
nice -n 10 $CMD 2>&1 | tee "$LOG_FILE"

# Check result
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ Training completed successfully!"
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    exit 0
else
    echo ""
    echo "========================================"
    echo "❌ Training failed!"
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Check log: $LOG_FILE"
    echo "========================================"
    exit 1
fi
