#!/bin/bash
#
# Run Level 1 Comparison for All Baselines
#
# This script runs all 3 rule-based baselines on Level 1
# (100 episodes, 20 satellites, ~2 hours total runtime)
#
# Usage:
#   ./scripts/run_level1_comparison.sh
#
# Output:
#   results/level1_comparison.csv
#   results/level1_comparison_report.md

set -e  # Exit on error

echo "============================================================"
echo "Level 1 Baseline Comparison"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Level: 1 (Quick Validation)"
echo "  Satellites: 20"
echo "  Episodes: 100"
echo "  Expected Time: ~2 hours"
echo ""

# Create results directory
mkdir -p results

# Run comparison
echo "Starting comparison..."
python scripts/evaluate_strategies.py \
    --compare-all \
    --level 1 \
    --episodes 100 \
    --seed 42 \
    --output results/level1_comparison.csv

echo ""
echo "============================================================"
echo "✅ Comparison Complete!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - results/level1_comparison.csv"
echo ""
echo "Next steps:"
echo "  1. Review comparison results"
echo "  2. Generate analysis report"
echo "  3. Compare with DQN (requires trained model)"
