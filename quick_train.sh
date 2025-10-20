#!/bin/bash
# Quick Training Script - 快速訓練腳本
# Usage: ./quick_train.sh [level]
# Example: ./quick_train.sh 1

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if level is provided
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}Usage: $0 [level]${NC}"
    echo ""
    echo "Available levels:"
    echo "  0 - Smoke Test (10分鐘) - 驗證系統"
    echo "  1 - Quick Test (2小時) - 驗證訓練邏輯"
    echo "  2 - Development (6小時) - 調試超參數"
    echo "  3 - Validation (10小時) - 驗證效果 ⭐ 推薦"
    echo "  4 - Baseline (21小時) - 建立基準"
    echo "  5 - Full Training (35小時) - 完整訓練"
    echo ""
    echo "Example: $0 3"
    exit 1
fi

LEVEL=$1

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Training configurations
case $LEVEL in
    0)
        echo -e "${GREEN}Starting Level 0: Smoke Test (10分鐘)${NC}"
        python3 train_online_rl.py \
            --num-satellites 10 \
            --num-episodes 10 \
            --output-dir output/level0_smoke_test \
            --log-interval 5
        ;;
    1)
        echo -e "${GREEN}Starting Level 1: Quick Test (2小時)${NC}"
        python3 train_online_rl.py \
            --num-satellites 20 \
            --num-episodes 100 \
            --output-dir output/level1_quick_test \
            --log-interval 10
        ;;
    2)
        echo -e "${GREEN}Starting Level 2: Development (6小時)${NC}"
        python3 train_online_rl.py \
            --num-satellites 50 \
            --num-episodes 300 \
            --output-dir output/level2_development \
            --log-interval 20 \
            --checkpoint-interval 100
        ;;
    3)
        echo -e "${GREEN}Starting Level 3: Validation (10小時) ⭐${NC}"
        python3 train_online_rl.py \
            --num-episodes 500 \
            --output-dir output/level3_validation \
            --log-interval 50 \
            --checkpoint-interval 100
        ;;
    4)
        echo -e "${GREEN}Starting Level 4: Baseline (21小時)${NC}"
        python3 train_online_rl.py \
            --num-episodes 1000 \
            --output-dir output/level4_baseline \
            --log-interval 50 \
            --checkpoint-interval 200
        ;;
    5)
        echo -e "${GREEN}Starting Level 5: Full Training (35小時)${NC}"
        echo -e "${YELLOW}警告: 這會跑很久！確定要繼續嗎? (y/n)${NC}"
        read -r confirm
        if [ "$confirm" != "y" ]; then
            echo -e "${RED}已取消${NC}"
            exit 0
        fi
        python3 train_online_rl.py \
            --num-episodes 1700 \
            --output-dir output/level5_full_training \
            --log-interval 100 \
            --checkpoint-interval 500
        ;;
    *)
        echo -e "${RED}Invalid level: $LEVEL${NC}"
        echo "Please choose 0-5"
        exit 1
        ;;
esac

echo -e "${GREEN}✅ Training completed!${NC}"
