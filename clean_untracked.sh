#!/bin/bash
# Clean all untracked files (simulates fresh clone)
# This removes all files that should be ignored by .gitignore

set -e

echo "=========================================="
echo "Clean Untracked Files"
echo "=========================================="
echo ""
echo "This script will remove all files that are NOT tracked by git."
echo "This simulates a fresh 'git clone' state."
echo ""

# Color codes
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Show what will be removed
echo -e "${YELLOW}Files and directories that will be removed:${NC}"
echo ""

# List untracked files (dry run)
if git clean -xdn | grep -q "Would remove"; then
    git clean -xdn
    echo ""
else
    echo "No untracked files to remove."
    echo ""
    exit 0
fi

# Ask for confirmation
echo -e "${RED}WARNING: This will permanently delete the above files!${NC}"
echo ""
read -p "Continue? (yes/no): " -r
echo

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Remove untracked files
echo ""
echo "Cleaning..."
git clean -xdf

echo ""
echo -e "${GREEN}✅ Cleanup complete!${NC}"
echo ""
echo "Repository is now in 'fresh clone' state."
echo "All tracked files are preserved."
echo "All untracked files have been removed."
