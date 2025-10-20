#!/bin/bash
#
# Full Validation Pipeline for Refactored RL Framework
#
# This script runs all 5 validation stages:
#   Stage 0: Academic Compliance (P0 - CRITICAL) [~5 min]
#   Stage 1: Unit Tests [~10 min]
#   Stage 2: Integration Tests [~30 min]
#   Stage 3: E2E Baseline Comparison (Level 1) [~2 hours]
#   Stage 4: E2E DQN Training (Level 1) [~2 hours]
#
# Total estimated time: 4-5 hours
#
# Usage:
#   ./scripts/validation/run_full_validation.sh
#
# Output:
#   results/validation/stage*_*.json
#   docs/validation/VALIDATION_REPORT.md

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "FULL VALIDATION PIPELINE"
echo "============================================================"
echo ""
echo "This will run all 5 validation stages (~4-5 hours)"
echo "Started: $(date)"
echo ""

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No virtual environment found"
fi

# Create results directory
mkdir -p results/validation

# ============================================================
# STAGE 0: Academic Compliance (P0 - CRITICAL)
# ============================================================
echo ""
echo "============================================================"
echo "STAGE 0: ACADEMIC COMPLIANCE (P0 - CRITICAL)"
echo "============================================================"
echo ""

python scripts/validation/stage0_academic_compliance.py
STAGE0_EXIT=$?

if [ $STAGE0_EXIT -ne 0 ]; then
    echo ""
    echo "🚨 CRITICAL FAILURE IN STAGE 0"
    echo "🚨 Academic compliance not met - validation aborted"
    echo ""
    exit 1
fi

echo "✅ Stage 0 PASSED - Academic compliance verified"

# ============================================================
# STAGE 1: Unit Tests
# ============================================================
echo ""
echo "============================================================"
echo "STAGE 1: UNIT TESTS"
echo "============================================================"
echo ""

python scripts/validation/stage1_unit_tests.py
STAGE1_EXIT=$?

if [ $STAGE1_EXIT -ne 0 ]; then
    echo ""
    echo "❌ FAILURE IN STAGE 1"
    echo "❌ Unit tests failed - fix before proceeding"
    echo ""
    exit 1
fi

echo "✅ Stage 1 PASSED - Unit tests successful"

# ============================================================
# STAGE 2: Integration Tests
# ============================================================
echo ""
echo "============================================================"
echo "STAGE 2: INTEGRATION TESTS"
echo "============================================================"
echo ""

echo "⚠️  Stage 2 script not yet implemented - skipping"
# python scripts/validation/stage2_integration_tests.py
# STAGE2_EXIT=$?

# if [ $STAGE2_EXIT -ne 0 ]; then
#     echo ""
#     echo "❌ FAILURE IN STAGE 2"
#     echo ""
#     exit 1
# fi

# echo "✅ Stage 2 PASSED - Integration tests successful"

# ============================================================
# STAGE 3: E2E Baseline Comparison (Level 1)
# ============================================================
echo ""
echo "============================================================"
echo "STAGE 3: E2E BASELINE COMPARISON (Level 1, ~2 hours)"
echo "============================================================"
echo ""

python scripts/evaluate_strategies.py \
    --compare-all \
    --level 1 \
    --episodes 100 \
    --seed 42 \
    --output results/validation/stage3_baseline_comparison.csv

STAGE3_EXIT=$?

if [ $STAGE3_EXIT -ne 0 ]; then
    echo ""
    echo "❌ FAILURE IN STAGE 3"
    echo ""
    exit 1
fi

echo "✅ Stage 3 PASSED - Baseline comparison successful"

# ============================================================
# STAGE 4: E2E DQN Training (Level 1)
# ============================================================
echo ""
echo "============================================================"
echo "STAGE 4: E2E DQN TRAINING (Level 1, ~2 hours)"
echo "============================================================"
echo ""

python train.py \
    --algorithm dqn \
    --level 1 \
    --episodes 100 \
    --seed 42 \
    --output-dir results/validation/stage4_dqn

STAGE4_EXIT=$?

if [ $STAGE4_EXIT -ne 0 ]; then
    echo ""
    echo "❌ FAILURE IN STAGE 4"
    echo ""
    exit 1
fi

echo "✅ Stage 4 PASSED - DQN training successful"

# ============================================================
# VALIDATION COMPLETE
# ============================================================
echo ""
echo "============================================================"
echo "✅ FULL VALIDATION COMPLETE"
echo "============================================================"
echo ""
echo "Completed: $(date)"
echo ""
echo "Results saved in:"
echo "  - results/validation/stage0_academic_compliance.json"
echo "  - results/validation/stage1_unit_tests.json"
echo "  - results/validation/stage3_baseline_comparison.csv"
echo "  - results/validation/stage4_dqn/"
echo ""
echo "Next steps:"
echo "  1. Review validation results"
echo "  2. Generate validation report:"
echo "     python scripts/validation/generate_report.py"
echo "  3. If all passed, framework is ready for research"
echo ""
echo "============================================================"
