#!/bin/bash
# Monitor 30-day precompute generation progress

LOG_FILE="/tmp/precompute_30days_optimized.log"
OUTPUT_FILE="data/orbit_precompute_30days_optimized.h5"

echo "============================================================"
echo "30天預計算表生成進度監控"
echo "============================================================"
echo ""

# Check if process is running
if pgrep -f "generate_orbit_precompute.*30days" > /dev/null; then
    echo "✅ 生成進程運行中"
else
    echo "⚠️  生成進程未運行"
fi

echo ""
echo "最新進度:"
tail -10 "$LOG_FILE" | grep -E "Satellites|sat/s|Writing|✅.*complete" | tail -5

echo ""
echo "輸出文件狀態:"
if [ -f "$OUTPUT_FILE" ]; then
    SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo "  文件大小: $SIZE"
    echo "  最後修改: $(stat -c %y "$OUTPUT_FILE" | cut -d'.' -f1)"
else
    echo "  尚未生成"
fi

echo ""
echo "============================================================"
echo "完整日誌: tail -f $LOG_FILE"
echo "============================================================"
