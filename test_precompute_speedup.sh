#!/bin/bash
# Test training speedup with precompute table

set -e

PRECOMPUTE_TABLE="data/orbit_precompute/orbit_precompute_7days.h5"
OUTPUT_DIR="output/speedup_test"
CONFIG="configs/config.yaml"
EPISODES=10
SEED=42

echo "========================================="
echo "Precompute Speedup Test"
echo "========================================="
echo ""

# Check if precompute table exists
if [ ! -f "$PRECOMPUTE_TABLE" ]; then
    echo "❌ Precompute table not found: $PRECOMPUTE_TABLE"
    exit 1
fi

# Get table size and info
TABLE_SIZE=$(ls -lh "$PRECOMPUTE_TABLE" | awk '{print $5}')
echo "📊 Precompute Table: $TABLE_SIZE"
echo ""

# Test 1: Real-time mode (baseline)
echo "🔬 Test 1: Real-time mode (baseline)"
echo "-----------------------------------"
rm -rf "$OUTPUT_DIR/realtime"
START_TIME=$(date +%s)

python train_sb3.py \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR/realtime" \
    --num-episodes $EPISODES \
    --seed $SEED \
    > /tmp/speedup_test_realtime.log 2>&1

END_TIME=$(date +%s)
REALTIME_DURATION=$((END_TIME - START_TIME))
REALTIME_FPS=$(grep "fps" /tmp/speedup_test_realtime.log | tail -1 | grep -oP '\d+(?=\s+it/s)' || echo "N/A")

echo "✅ Completed in ${REALTIME_DURATION}s"
echo "   FPS: $REALTIME_FPS"
echo ""

# Test 2: Precompute mode
echo "🚀 Test 2: Precompute mode"
echo "-----------------------------------"

# Update config to use precompute table
cat > /tmp/config_with_precompute.yaml << YAML_EOF
# Inherit from main config
$(cat "$CONFIG")

# Override precompute settings
precompute:
  enabled: true
  table_path: "$PRECOMPUTE_TABLE"
YAML_EOF

rm -rf "$OUTPUT_DIR/precompute"
START_TIME=$(date +%s)

python train_sb3.py \
    --config /tmp/config_with_precompute.yaml \
    --output-dir "$OUTPUT_DIR/precompute" \
    --num-episodes $EPISODES \
    --seed $SEED \
    > /tmp/speedup_test_precompute.log 2>&1

END_TIME=$(date +%s)
PRECOMPUTE_DURATION=$((END_TIME - START_TIME))
PRECOMPUTE_FPS=$(grep "fps" /tmp/speedup_test_precompute.log | tail -1 | grep -oP '\d+(?=\s+it/s)' || echo "N/A")

echo "✅ Completed in ${PRECOMPUTE_DURATION}s"
echo "   FPS: $PRECOMPUTE_FPS"
echo ""

# Calculate speedup
if [ "$REALTIME_DURATION" -gt 0 ] && [ "$PRECOMPUTE_DURATION" -gt 0 ]; then
    SPEEDUP=$(echo "scale=1; $REALTIME_DURATION / $PRECOMPUTE_DURATION" | bc)
    echo "========================================="
    echo "📊 Results Summary"
    echo "========================================="
    echo "Real-time mode:  ${REALTIME_DURATION}s (FPS: $REALTIME_FPS)"
    echo "Precompute mode: ${PRECOMPUTE_DURATION}s (FPS: $PRECOMPUTE_FPS)"
    echo "Speedup:         ${SPEEDUP}x"
    echo ""
    
    if (( $(echo "$SPEEDUP >= 50" | bc -l) )); then
        echo "✅ Target speedup achieved (≥50x)"
    elif (( $(echo "$SPEEDUP >= 10" | bc -l) )); then
        echo "⚠️  Significant speedup but below target (10-50x)"
    else
        echo "❌ Speedup below expectation (<10x)"
    fi
else
    echo "❌ Could not calculate speedup"
fi

echo "========================================="
