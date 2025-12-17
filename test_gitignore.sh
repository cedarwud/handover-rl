#!/bin/bash
# Test .gitignore configuration for simplified architecture

set -e

echo "=========================================="
echo "Testing .gitignore Configuration"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

PASS_COUNT=0
FAIL_COUNT=0

# Helper function
test_tracked() {
    local file=$1
    local description=$2

    if git check-ignore -q "$file" 2>/dev/null; then
        echo -e "${RED}❌ FAIL${NC}: $description"
        echo "   File: $file (should be tracked, but is ignored)"
        ((FAIL_COUNT++))
    else
        echo -e "${GREEN}✅ PASS${NC}: $description"
        ((PASS_COUNT++))
    fi
}

test_ignored() {
    local file=$1
    local description=$2

    if git check-ignore -q "$file" 2>/dev/null; then
        echo -e "${GREEN}✅ PASS${NC}: $description"
        ((PASS_COUNT++))
    else
        echo -e "${RED}❌ FAIL${NC}: $description"
        echo "   File: $file (should be ignored, but is tracked)"
        ((FAIL_COUNT++))
    fi
}

# Create test files structure
echo "Creating test file structure..."
mkdir -p data/satellite_pool
mkdir -p data/orbit_precompute
mkdir -p output/test
mkdir -p logs

# Create test files
touch data/satellite_pool/snapshot_v1.0.json
touch data/satellite_pool/snapshot_v1.0.metadata.json
touch data/satellite_pool/.gitkeep
touch data/satellite_pool/README.md
touch data/orbit_precompute/.gitkeep
touch data/orbit_precompute/README.md
touch data/orbit_precompute/test_precompute.h5
touch data/orbit_precompute/orbit_precompute_30days.h5
touch data/satellite_ids_from_precompute.txt
touch output/test/model.zip
touch logs/training.log

echo ""
echo "=========================================="
echo "Test Results"
echo "=========================================="
echo ""

# Test 1: Satellite pool snapshots should be TRACKED
echo "Test Group 1: Satellite Pool Snapshots (should be tracked)"
echo "----------------------------------------------------------"
test_tracked "data/satellite_pool/snapshot_v1.0.json" "Snapshot JSON"
test_tracked "data/satellite_pool/snapshot_v1.0.metadata.json" "Snapshot metadata"
test_tracked "data/satellite_pool/.gitkeep" "Satellite pool .gitkeep"
test_tracked "data/satellite_pool/README.md" "Satellite pool README"
echo ""

# Test 2: Precompute structure should be TRACKED, but .h5 files IGNORED
echo "Test Group 2: Orbit Precompute (structure tracked, .h5 ignored)"
echo "----------------------------------------------------------------"
test_tracked "data/orbit_precompute/.gitkeep" "Precompute .gitkeep"
test_tracked "data/orbit_precompute/README.md" "Precompute README"
test_ignored "data/orbit_precompute/test_precompute.h5" "Precompute test.h5"
test_ignored "data/orbit_precompute/orbit_precompute_30days.h5" "Precompute 30days.h5"
echo ""

# Test 3: Legacy files should be TRACKED
echo "Test Group 3: Legacy Files (for compatibility)"
echo "-----------------------------------------------"
test_tracked "data/satellite_ids_from_precompute.txt" "Legacy satellite IDs"
echo ""

# Test 4: Large outputs should be IGNORED
echo "Test Group 4: Training Outputs (should be ignored)"
echo "---------------------------------------------------"
test_ignored "output/test/model.zip" "Training model output"
test_ignored "logs/training.log" "Training log"
echo ""

# Cleanup
echo "Cleaning up test files..."
rm -f data/satellite_pool/snapshot_v1.0.json
rm -f data/satellite_pool/snapshot_v1.0.metadata.json
rm -f data/satellite_pool/.gitkeep
rm -f data/satellite_pool/README.md
rm -f data/orbit_precompute/.gitkeep
rm -f data/orbit_precompute/README.md
rm -f data/orbit_precompute/test_precompute.h5
rm -f data/orbit_precompute/orbit_precompute_30days.h5
rm -f data/satellite_ids_from_precompute.txt
rm -f output/test/model.zip
rm -f logs/training.log
rmdir output/test 2>/dev/null || true
rmdir data/satellite_pool 2>/dev/null || true
rmdir data/orbit_precompute 2>/dev/null || true

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo -e "${GREEN}✅ Passed: $PASS_COUNT${NC}"
echo -e "${RED}❌ Failed: $FAIL_COUNT${NC}"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed! .gitignore is correctly configured.${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Some tests failed. Please review .gitignore configuration.${NC}"
    exit 1
fi
