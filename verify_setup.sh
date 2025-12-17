#!/bin/bash
# Verify handover-rl setup (following README.md instructions)

set -e

echo "=========================================="
echo "Handover-RL Setup Verification"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS_COUNT=0
FAIL_COUNT=0

# Helper functions
check_pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
    ((PASS_COUNT++))
}

check_fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    echo -e "${YELLOW}   $2${NC}"
    ((FAIL_COUNT++))
}

check_warn() {
    echo -e "${YELLOW}⚠️  WARN${NC}: $1"
    echo -e "${YELLOW}   $2${NC}"
}

echo "Step 1: Check orbit-engine dependency"
echo "--------------------------------------"

ORBIT_ENGINE_DIR="/home/sat/satellite/orbit-engine"
STAGE4_DIR="$ORBIT_ENGINE_DIR/data/outputs/stage4"

if [ -d "$ORBIT_ENGINE_DIR" ]; then
    check_pass "orbit-engine directory exists"

    if [ -d "$STAGE4_DIR" ]; then
        STAGE4_FILES=$(ls $STAGE4_DIR/link_feasibility_output_*.json 2>/dev/null | wc -l)

        if [ "$STAGE4_FILES" -gt 0 ]; then
            LATEST_FILE=$(ls -t $STAGE4_DIR/link_feasibility_output_*.json | head -1)
            FILE_SIZE=$(du -h "$LATEST_FILE" | cut -f1)
            FILE_DATE=$(stat -c %y "$LATEST_FILE" | cut -d' ' -f1)

            check_pass "orbit-engine Stage 4 output found"
            echo "   File: $(basename $LATEST_FILE)"
            echo "   Size: $FILE_SIZE"
            echo "   Date: $FILE_DATE"

            # Check data freshness (TLE age)
            DAYS_OLD=$(( ($(date +%s) - $(stat -c %Y "$LATEST_FILE")) / 86400 ))
            if [ "$DAYS_OLD" -le 14 ]; then
                check_pass "Data freshness: $DAYS_OLD days old (recommended: ≤14 days)"
            elif [ "$DAYS_OLD" -le 30 ]; then
                check_warn "Data age: $DAYS_OLD days old" \
                           "Recommended to regenerate Stage 4 (target: ≤14 days)"
            else
                check_fail "Data too old: $DAYS_OLD days" \
                           "Please run: cd $ORBIT_ENGINE_DIR && ./run.sh --stage 4"
            fi
        else
            check_fail "orbit-engine Stage 4 output missing" \
                       "Run: cd $ORBIT_ENGINE_DIR && ./run.sh --stage 4"
        fi
    else
        check_fail "Stage 4 directory missing" \
                   "Run: cd $ORBIT_ENGINE_DIR && ./run.sh --stage 4"
    fi
else
    check_fail "orbit-engine not found at $ORBIT_ENGINE_DIR" \
               "Clone orbit-engine as sibling directory"
fi

echo ""
echo "Step 2: Check Python environment"
echo "---------------------------------"

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | grep -oP '\d+\.\d+')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        check_pass "Python version: $PYTHON_VERSION (required: ≥3.10)"
    else
        check_fail "Python version: $PYTHON_VERSION" \
                   "Required: Python 3.10 or higher"
    fi
else
    check_fail "Python 3 not found" "Install Python 3.10+"
fi

if [ -d "venv" ]; then
    check_pass "Virtual environment exists"

    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate

        # Check key dependencies
        if python -c "import torch" 2>/dev/null; then
            TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
            check_pass "PyTorch installed: $TORCH_VERSION"
        else
            check_warn "PyTorch not installed" "May need to run: pip install -r requirements.txt"
        fi

        if python -c "import stable_baselines3" 2>/dev/null; then
            SB3_VERSION=$(python -c "import stable_baselines3; print(stable_baselines3.__version__)")
            check_pass "Stable-Baselines3 installed: $SB3_VERSION"
        else
            check_warn "Stable-Baselines3 not installed" "May need to run: pip install -r requirements.txt"
        fi

        if python -c "import gymnasium" 2>/dev/null; then
            GYM_VERSION=$(python -c "import gymnasium; print(gymnasium.__version__)")
            check_pass "Gymnasium installed: $GYM_VERSION"
        else
            check_warn "Gymnasium not installed" "May need to run: pip install -r requirements.txt"
        fi

        deactivate
    fi
else
    check_warn "Virtual environment not found" "Run: ./setup_env.sh"
fi

echo ""
echo "Step 3: Check handover-rl data files"
echo "-------------------------------------"

if [ -f "data/orbit_precompute/orbit_precompute_30days.h5" ]; then
    PRECOMPUTE_SIZE=$(du -h data/orbit_precompute/orbit_precompute_30days.h5 | cut -f1)
    check_pass "Precompute table exists: $PRECOMPUTE_SIZE"

    # Check if precompute table is fresh
    PRECOMPUTE_DAYS_OLD=$(( ($(date +%s) - $(stat -c %Y "data/orbit_precompute/orbit_precompute_30days.h5")) / 86400 ))

    if [ "$PRECOMPUTE_DAYS_OLD" -le 30 ]; then
        check_pass "Precompute table age: $PRECOMPUTE_DAYS_OLD days (acceptable)"
    else
        check_warn "Precompute table old: $PRECOMPUTE_DAYS_OLD days" \
                   "Consider regenerating with fresh orbit-engine data"
    fi
else
    check_warn "Precompute table not found" \
               "Training will use real-time calculation (100-1000x slower)"
    echo "   Generate with: python tools/orbit/generate_orbit_precompute.py"
fi

if [ -f "data/satellite_ids_from_precompute.txt" ]; then
    SAT_COUNT=$(wc -l < data/satellite_ids_from_precompute.txt)
    check_pass "Satellite IDs file exists: $SAT_COUNT satellites"
else
    check_warn "Satellite IDs file not found" "Will be auto-generated if needed"
fi

echo ""
echo "Step 4: Check directory structure"
echo "----------------------------------"

for dir in "configs" "src" "scripts" "tests" "data" "output" "logs"; do
    if [ -d "$dir" ]; then
        check_pass "Directory exists: $dir/"
    else
        check_fail "Directory missing: $dir/" "Repository may be corrupted"
    fi
done

if [ -f "train_sb3.py" ]; then
    check_pass "Main training script: train_sb3.py"
else
    check_fail "train_sb3.py missing" "Repository may be corrupted"
fi

if [ -f "configs/config.yaml" ]; then
    check_pass "Configuration file: configs/config.yaml"
else
    check_fail "configs/config.yaml missing" "Repository may be corrupted"
fi

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo -e "${GREEN}✅ Passed: $PASS_COUNT${NC}"
echo -e "${RED}❌ Failed: $FAIL_COUNT${NC}"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}🎉 All critical checks passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Activate environment: source venv/bin/activate"
    echo "  2. Quick test: python train_sb3.py --config configs/config.yaml --num-episodes 100"
    echo "  3. Full training: python train_sb3.py --config configs/config.yaml --num-episodes 2500 --seed 42"
    echo ""
    exit 0
else
    echo -e "${RED}⚠️  Some checks failed. Please resolve issues above.${NC}"
    echo ""
    exit 1
fi
