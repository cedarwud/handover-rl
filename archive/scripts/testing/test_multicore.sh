#!/bin/bash
# Multi-core Training Test Script
# Tests the new vectorized environment implementation

set -e  # Exit on error

echo "=========================================="
echo "多核心訓練測試腳本"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test configurations
echo -e "${BLUE}測試配置:${NC}"
echo "  工作目錄: $(pwd)"
echo "  Python: $(./venv/bin/python --version)"
echo ""

# Function to run test
run_test() {
    local num_envs=$1
    local test_name=$2

    echo -e "${YELLOW}=========================================${NC}"
    echo -e "${YELLOW}測試: $test_name (${num_envs} 核心)${NC}"
    echo -e "${YELLOW}=========================================${NC}"

    time ./venv/bin/python train.py \
        --algorithm dqn \
        --level 0 \
        --config config/epsilon_fixed_config.yaml \
        --output-dir output/test_${num_envs}cores \
        --num-envs ${num_envs} \
        --seed 42

    echo -e "${GREEN}✅ 測試完成: $test_name${NC}"
    echo ""
}

# Test 1: Single core (baseline)
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}測試 1: 單核心基準測試${NC}"
echo -e "${BLUE}=========================================${NC}"
run_test 1 "單核心基準"

# Test 2: 8 cores
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}測試 2: 8 核心測試${NC}"
echo -e "${BLUE}=========================================${NC}"
run_test 8 "8 核心"

# Test 3: 30 cores (if successful, comment out to save time)
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}測試 3: 30 核心測試${NC}"
echo -e "${BLUE}=========================================${NC}"
run_test 30 "30 核心"

# Summary
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}所有測試完成！${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "檢查結果:"
echo "  output/test_1cores/  - 單核心基準"
echo "  output/test_8cores/  - 8 核心"
echo "  output/test_30cores/ - 30 核心"
echo ""
echo "比較訓練時間來驗證加速效果"
