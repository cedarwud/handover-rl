#!/bin/bash
# 實時訓練儀表板

while true; do
    clear
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║          Level 5 訓練實時監控 - 每30秒更新                 ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    date
    echo ""

    # Level 5 進度
    echo "=== Level 5 主訓練 (1700 episodes) ==="
    PROGRESS=$(grep "Training:" training_level5_20min_final.log | tail -1)
    echo "$PROGRESS"

    # 提取完成數量
    COMPLETED=$(echo "$PROGRESS" | grep -oP '\d+(?=/1700)' | head -1)
    if [ -n "$COMPLETED" ]; then
        PERCENT=$(echo "scale=2; $COMPLETED * 100 / 1700" | bc)
        echo "進度: $COMPLETED / 1700 ($PERCENT%)"
    fi
    echo ""

    # 驗證測試進度
    echo "=== 驗證測試 (50 episodes) ==="
    TEST_PROGRESS=$(grep "Training:" test_20min_config.log 2>/dev/null | tail -1)
    echo "$TEST_PROGRESS"
    echo ""

    # 系統資源
    echo "=== 系統資源 ==="
    PYTHON_PROCS=$(ps aux | grep "python3.*train.py" | grep -v grep | wc -l)
    echo "Python 進程數: $PYTHON_PROCS"

    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    echo "總 CPU 使用: ${CPU_USAGE}%"

    MEM_USAGE=$(free -h | awk '/^Mem:/ {print $3 "/" $2}')
    echo "記憶體使用: $MEM_USAGE"
    echo ""

    # 數值穩定性
    echo "=== 數值穩定性檢查 ==="
    NAN_COUNT=$(grep -c "\[NaN\|\[Inf" training_level5_20min_final.log 2>/dev/null || echo 0)
    if [ "$NAN_COUNT" -eq 0 ]; then
        echo "✅ 無 NaN/Inf 問題"
    else
        echo "⚠️  發現 $NAN_COUNT 次 NaN/Inf 警告"
    fi

    # Log 大小
    LOG_SIZE=$(ls -lh training_level5_20min_final.log 2>/dev/null | awk '{print $5}')
    echo "Log 文件大小: $LOG_SIZE"
    echo ""

    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  按 Ctrl+C 退出                                              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    sleep 30
done
