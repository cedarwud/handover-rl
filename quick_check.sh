#!/bin/bash
# 快速檢查腳本 - 隨時手動執行

LOG_FILE="${1:-long_training_17k.log}"

echo "========================================="
echo "🔍 訓練狀態快速檢查"
echo "========================================="
echo "日誌文件: $LOG_FILE"
echo ""

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ 日誌文件不存在"
    exit 1
fi

# 1. 當前進度
echo "【1. 當前進度】"
LATEST=$(grep "Episode" "$LOG_FILE" | tail -1)
echo "$LATEST"
echo ""

# 2. 錯誤檢查
echo "【2. 錯誤統計】"
NAN_COUNT=$(grep -c "NaN/Inf Detection" "$LOG_FILE" || echo "0")
LARGE_LOSS=$(grep -c "Large Loss Warning" "$LOG_FILE" || echo "0")
echo "  NaN/Inf 錯誤: $NAN_COUNT"
echo "  Large Loss 警告: $LARGE_LOSS"
echo ""

# 3. Episode 920 檢查
echo "【3. Episode 920 檢查】"
EP920=$(grep "Episode  920" "$LOG_FILE" 2>/dev/null)
if [ -n "$EP920" ]; then
    echo "  ✅ 已通過 Episode 920:"
    echo "  $EP920"
else
    CURRENT_EP=$(grep "Episode" "$LOG_FILE" | tail -1 | grep -oP 'Episode\s+\K[0-9]+' || echo "0")
    if [ "$CURRENT_EP" -lt 920 ]; then
        echo "  ⏳ 尚未到達 Episode 920 (當前: $CURRENT_EP)"
    else
        echo "  ✅ 已通過 Episode 920"
    fi
fi
echo ""

# 4. 最近10個 episodes
echo "【4. 最近10個 episodes】"
grep "Episode.*reward=" "$LOG_FILE" | tail -10
echo ""

# 5. GPU 狀態
echo "【5. GPU 狀態】"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader
else
    echo "  nvidia-smi 不可用"
fi
echo ""

# 6. 訓練時間估算
echo "【6. 訓練時間估算】"
CURRENT_EP=$(grep "Episode" "$LOG_FILE" | tail -1 | grep -oP 'Episode\s+\K[0-9]+' || echo "0")
if [ "$CURRENT_EP" -gt 0 ]; then
    # 獲取訓練開始時間
    START_LINE=$(grep "Starting training" "$LOG_FILE" | head -1)
    if [ -n "$START_LINE" ]; then
        PROGRESS=$(echo "scale=2; $CURRENT_EP * 100 / 17000" | bc)
        REMAINING=$(echo "17000 - $CURRENT_EP" | bc)
        echo "  已完成: $CURRENT_EP/17000 ($PROGRESS%)"
        echo "  剩餘: $REMAINING episodes"
        
        # 估算剩餘時間（假設 22秒/episode）
        REMAINING_SECONDS=$(echo "$REMAINING * 22" | bc)
        REMAINING_HOURS=$(echo "scale=1; $REMAINING_SECONDS / 3600" | bc)
        echo "  預估剩餘時間: $REMAINING_HOURS 小時"
    fi
fi
echo ""

# 7. 最新 checkpoint
echo "【7. 最新 checkpoint】"
CHECKPOINT_DIR="output/long_training_17k/checkpoints"
if [ -d "$CHECKPOINT_DIR" ]; then
    ls -lth "$CHECKPOINT_DIR"/checkpoint_ep*.pth 2>/dev/null | head -5 || echo "  沒有 checkpoint 文件"
else
    echo "  Checkpoint 目錄不存在"
fi
echo ""

echo "========================================="
echo "✅ 檢查完成"
echo "========================================="
