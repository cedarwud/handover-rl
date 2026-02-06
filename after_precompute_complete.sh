#!/bin/bash
# Actions to perform after precompute table generation completes

set -e

PRECOMPUTE_TABLE="data/orbit_precompute/orbit_precompute_7days.h5"

echo "========================================="
echo "預計算表生成完成後處理"
echo "========================================="
echo ""

# Step 1: Verify HDF5 file
echo "1️⃣  驗證 HDF5 文件..."
if [ ! -f "$PRECOMPUTE_TABLE" ]; then
    echo "❌ 文件不存在: $PRECOMPUTE_TABLE"
    exit 1
fi

FILE_SIZE=$(ls -lh "$PRECOMPUTE_TABLE" | awk '{print $5}')
echo "   文件大小: $FILE_SIZE"

# Use Python to validate HDF5 structure
python << PYTHON_EOF
import h5py
import sys

try:
    with h5py.File("$PRECOMPUTE_TABLE", 'r') as f:
        num_sats = f['metadata'].attrs['num_satellites']
        num_timesteps = f['metadata'].attrs['num_timesteps']
        print(f"   衛星數量: {num_sats}")
        print(f"   時間步數: {num_timesteps}")
        print(f"   總狀態數: {num_sats * num_timesteps:,}")
        print("   ✅ HDF5 結構驗證通過")
        sys.exit(0)
except Exception as e:
    print(f"   ❌ HDF5 驗證失敗: {e}")
    sys.exit(1)
PYTHON_EOF

if [ $? -ne 0 ]; then
    echo "❌ HDF5 驗證失敗"
    exit 1
fi

echo ""

# Step 2: Test training speedup
echo "2️⃣  測試訓練加速效果..."
echo "   執行: ./test_precompute_speedup.sh"
echo ""

./test_precompute_speedup.sh

echo ""

# Step 3: Update config for future use
echo "3️⃣  更新配置文件..."
echo "   建議手動更新 configs/config.yaml:"
echo "   precompute:"
echo "     enabled: true"
echo "     table_path: \"$PRECOMPUTE_TABLE\""
echo ""

# Step 4: Summary
echo "========================================="
echo "✅ 預計算表已就緒！"
echo "========================================="
echo "下一步："
echo "  - 如果 7 天測試成功，生成 30 天完整表"
echo "  - 執行多種子訓練實驗"
echo "  - 收集論文實驗數據"
echo ""

