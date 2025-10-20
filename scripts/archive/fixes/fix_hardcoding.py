#!/usr/bin/env python3
"""
Fix Hardcoding Issues

Scans all Python files and reports hardcoded satellite IDs

Academic Compliance Check:
- NO HARDCODING violations
- Report all instances for manual review
"""

import re
from pathlib import Path
from typing import List, Tuple


def find_hardcoded_satellites(file_path: Path) -> List[Tuple[int, str]]:
    """
    Find lines with hardcoded STARLINK satellite IDs

    Returns:
        List of (line_number, line_content) tuples
    """
    violations = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, start=1):
            # Look for hardcoded STARLINK IDs
            if re.search(r'"STARLINK-\d+"', line):
                # Exclude acceptable cases (comments, documentation)
                if 'SOURCE:' in line or 'Example:' in line or '#' in line:
                    continue

                # Exclude docstring examples (e.g., "STARLINK-1008")
                if 'e.g.,' in line or '(e.g.,' in line:
                    continue

                # Exclude utility functions that define the pattern
                if 'startswith' in line or 'prefix' in line:
                    continue

                violations.append((line_num, line.strip()))

    return violations


def scan_project(root_dir: Path, exclude_dirs: List[str] = None):
    """
    Scan entire project for hardcoding violations
    """
    if exclude_dirs is None:
        exclude_dirs = ['venv', '.git', '__pycache__', 'output', 'data', 'orbit-engine']

    print("=" * 80)
    print("🔍 Scanning for Hardcoded Satellite IDs (Academic Compliance Check)")
    print("=" * 80)

    total_files = 0
    total_violations = 0
    files_with_violations = []

    for py_file in root_dir.rglob('*.py'):
        # Skip excluded directories
        if any(excluded in str(py_file) for excluded in exclude_dirs):
            continue

        total_files += 1
        violations = find_hardcoded_satellites(py_file)

        if violations:
            files_with_violations.append((py_file, violations))
            total_violations += len(violations)

    # Print results
    print(f"\n📊 Scan Results:")
    print(f"   Files scanned: {total_files}")
    print(f"   Files with violations: {len(files_with_violations)}")
    print(f"   Total violations: {total_violations}")

    if files_with_violations:
        print(f"\n❌ HARDCODING VIOLATIONS FOUND:\n")

        for file_path, violations in files_with_violations:
            rel_path = file_path.relative_to(root_dir)
            print(f"\n📁 {rel_path}")
            print(f"   {len(violations)} violation(s):")

            for line_num, line_content in violations[:3]:  # Show first 3
                print(f"      Line {line_num}: {line_content[:70]}...")

            if len(violations) > 3:
                print(f"      ... and {len(violations) - 3} more")

        print(f"\n{'=' * 80}")
        print("⚠️  ACADEMIC COMPLIANCE: FAILED")
        print("   Action required: Replace all hardcoded IDs with TLE extraction")
        print("   Use: from utils.satellite_utils import load_satellite_ids")
        print("=" * 80)
        return False
    else:
        print(f"\n{'=' * 80}")
        print("✅ ACADEMIC COMPLIANCE: PASSED")
        print("   No hardcoded satellite IDs found")
        print("=" * 80)
        return True


if __name__ == "__main__":
    project_root = Path(__file__).parent
    success = scan_project(project_root)
    exit(0 if success else 1)
