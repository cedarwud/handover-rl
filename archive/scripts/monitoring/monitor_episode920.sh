#!/bin/bash
# 監控 Episode 920 是否出現 loss 爆炸

LOG_FILE="training_level5_20min_final.log"

echo "========================================"
echo "Episode 920 Bug 監控"
echo "========================================"
echo ""

# 檢查當前進度
CURRENT=$(grep "Training:" $LOG_FILE | tail -1 | grep -oP '\d+(?=/1700)' | head -1)
echo "當前進度: Episode $CURRENT / 1700"
echo ""

if [ -z "$CURRENT" ]; then
    echo "⏳ 訓練剛開始，還沒有進度數據"
    exit 0
fi

# 檢查是否已經過 Episode 920
if [ "$CURRENT" -ge 920 ]; then
    echo "✅ 已通過 Episode 920！"
    echo ""

    # 查找 Episode 920 的記錄
    echo "=== Episode 920 附近的數據 ==="
    grep -E "Episode.*(91[0-9]|92[0-9]).*reward=" $LOG_FILE | tail -20

    echo ""
    echo "=== Episode 920 的 loss 檢查 ==="
    EP920_LOSS=$(grep "Episode.*920.*loss=" $LOG_FILE | grep -oP 'loss=\K[0-9.]+' | head -1)

    if [ -n "$EP920_LOSS" ]; then
        echo "Episode 920 loss: $EP920_LOSS"

        # 檢查是否爆炸（> 1000）
        if (( $(echo "$EP920_LOSS > 1000" | bc -l) )); then
            echo "❌ LOSS 爆炸！loss = $EP920_LOSS"
        else
            echo "✅ Loss 正常（< 1000）"
        fi
    else
        echo "⏳ Episode 920 數據還未記錄"
    fi
else
    echo "⏳ 還需要 $((920 - CURRENT)) episodes 到達 Episode 920"
    HOURS_TO_920=$(echo "scale=1; (920 - $CURRENT) * 3 / 60" | bc)
    echo "預計時間: ~$HOURS_TO_920 小時"
fi

echo ""
echo "========================================"
echo "提示: 當進度 >= 920 時重新運行此腳本查看結果"
echo "========================================"
