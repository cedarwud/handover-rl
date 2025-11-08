#!/bin/bash
# 檢查訓練里程碑並記錄

LOG_FILE="training_level5_20min_final.log"
MILESTONE_FILE="training_milestones.txt"

# 初始化里程碑文件
if [ ! -f "$MILESTONE_FILE" ]; then
    echo "訓練里程碑記錄" > $MILESTONE_FILE
    echo "開始時間: $(date)" >> $MILESTONE_FILE
    echo "==================" >> $MILESTONE_FILE
fi

# 獲取當前進度
CURRENT=$(grep "Training:" $LOG_FILE | tail -1 | grep -oP '\d+(?=/1700)' | head -1)

if [ -z "$CURRENT" ]; then
    exit 0
fi

# 定義里程碑
MILESTONES=(100 200 500 920 1000 1500 1700)

for MILESTONE in "${MILESTONES[@]}"; do
    # 檢查是否已達到且未記錄
    if [ "$CURRENT" -ge "$MILESTONE" ]; then
        if ! grep -q "Episode $MILESTONE" $MILESTONE_FILE; then
            echo "" >> $MILESTONE_FILE
            echo "✅ Episode $MILESTONE 完成" >> $MILESTONE_FILE
            echo "   時間: $(date)" >> $MILESTONE_FILE

            # 如果是 Episode 920，特別記錄
            if [ "$MILESTONE" -eq 920 ]; then
                EP920_DATA=$(grep "Episode.*920.*reward=" $LOG_FILE | tail -1)
                echo "   數據: $EP920_DATA" >> $MILESTONE_FILE

                EP920_LOSS=$(echo "$EP920_DATA" | grep -oP 'loss=\K[0-9.]+')
                if [ -n "$EP920_LOSS" ]; then
                    if (( $(echo "$EP920_LOSS < 10" | bc -l) )); then
                        echo "   ✅ Loss 正常: $EP920_LOSS" >> $MILESTONE_FILE
                    else
                        echo "   ⚠️  Loss 異常: $EP920_LOSS" >> $MILESTONE_FILE
                    fi
                fi
            fi

            echo "🎯 里程碑達成: Episode $MILESTONE"
        fi
    fi
done

# 顯示當前進度
echo "當前進度: $CURRENT / 1700"
echo ""
echo "已達成的里程碑:"
cat $MILESTONE_FILE
