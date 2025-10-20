#!/bin/bash
# Dependency Checker for handover-rl
#
# Purpose: Verify that orbit-engine (sibling repository) is present
#          and all required algorithm modules are available.
#
# Usage: ./scripts/setup/check_dependencies.sh

set -e

HANDOVER_RL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORBIT_ENGINE_ROOT="$(dirname "$HANDOVER_RL_ROOT")/orbit-engine"

echo "========================================================================"
echo "handover-rl Dependency Checker"
echo "========================================================================"
echo ""
echo "Checking runtime dependencies..."
echo ""

# Check 1: orbit-engine directory exists
echo "Check 1: orbit-engine (sibling repository)"
echo "Expected location: $ORBIT_ENGINE_ROOT"

if [ ! -d "$ORBIT_ENGINE_ROOT" ]; then
    echo "❌ FAILED: orbit-engine not found!"
    echo ""
    echo "   handover-rl requires orbit-engine as a sibling directory for algorithm modules."
    echo ""
    echo "   Expected structure:"
    echo "   parent_directory/"
    echo "   ├─ orbit-engine/      ← Required dependency (Git repo 1)"
    echo "   ├─ handover-rl/       ← Current directory (Git repo 2)"
    echo "   └─ tle_data/          ← Shared TLE data (Git repo 3)"
    echo ""
    echo "   To fix:"
    echo "   1. Clone orbit-engine:"
    echo "      cd $(dirname "$HANDOVER_RL_ROOT")"
    echo "      git clone https://github.com/your-org/orbit-engine.git"
    echo ""
    echo "   2. Re-run this script to verify."
    echo ""
    exit 1
else
    echo "✅ PASSED: orbit-engine found"
    echo ""
fi

# Check 2: Required orbit-engine modules
echo "Check 2: orbit-engine algorithm modules"

REQUIRED_MODULES=(
    "src/stages/stage2_orbital_computing/sgp4_calculator.py"
    "src/stages/stage5_signal_analysis/itur_physics_calculator.py"
    "src/stages/stage5_signal_analysis/gpp_ts38214_signal_calculator.py"
    "src/stages/stage5_signal_analysis/itur_official_atmospheric_model.py"
)

ALL_MODULES_OK=true

for module in "${REQUIRED_MODULES[@]}"; do
    if [ ! -f "$ORBIT_ENGINE_ROOT/$module" ]; then
        echo "❌ FAILED: Required module not found: $module"
        ALL_MODULES_OK=false
    else
        echo "✅ Found: $module"
    fi
done

if [ "$ALL_MODULES_OK" = false ]; then
    echo ""
    echo "❌ Some required modules are missing!"
    echo "   Please ensure orbit-engine is up-to-date:"
    echo "   cd $ORBIT_ENGINE_ROOT"
    echo "   git pull"
    exit 1
fi

echo "✅ PASSED: All required modules found"
echo ""

# Check 3: Python can import orbit-engine modules
echo "Check 3: Python import test"

python3 << 'PYTEST'
import sys
from pathlib import Path

# Get orbit-engine path
handover_rl_root = Path(__file__).resolve().parent
orbit_engine_root = handover_rl_root.parent / "orbit-engine"

# Try to import
try:
    sys.path.insert(0, str(orbit_engine_root))

    # Test imports (same as OrbitEngineAdapter)
    import os
    _cwd = os.getcwd()
    os.chdir(orbit_engine_root)

    from src.stages.stage2_orbital_computing.sgp4_calculator import SGP4Calculator
    from src.stages.stage5_signal_analysis.itur_physics_calculator import create_itur_physics_calculator
    from src.stages.stage5_signal_analysis.gpp_ts38214_signal_calculator import create_3gpp_signal_calculator
    from src.stages.stage5_signal_analysis.itur_official_atmospheric_model import create_itur_official_model

    os.chdir(_cwd)

    print("✅ PASSED: Python imports successful")
    sys.exit(0)

except ImportError as e:
    print(f"❌ FAILED: Python import error")
    print(f"   Error: {e}")
    print()
    print("   This may indicate:")
    print("   1. orbit-engine has missing dependencies")
    print("   2. orbit-engine source structure changed")
    print("   3. Python environment issues")
    print()
    print("   Try:")
    print(f"   cd {orbit_engine_root}")
    print("   pip install -r requirements.txt")
    sys.exit(1)
PYTEST

if [ $? -ne 0 ]; then
    exit 1
fi
echo ""

# Check 4: TLE data directory
echo "Check 4: TLE data (shared repository)"

# Load .env if exists
if [ -f "$HANDOVER_RL_ROOT/.env" ]; then
    export $(grep -v '^#' "$HANDOVER_RL_ROOT/.env" | xargs)
fi

TLE_DIR="${SATELLITE_TLE_DATA_DIR:-../tle_data}"

# Convert relative to absolute
if [[ "$TLE_DIR" != /* ]]; then
    TLE_DIR="$HANDOVER_RL_ROOT/$TLE_DIR"
fi

echo "Expected location: $TLE_DIR"

if [ ! -d "$TLE_DIR" ]; then
    echo "⚠️  WARNING: TLE data directory not found"
    echo ""
    echo "   This is not critical for handover-rl development, but required for evaluation."
    echo "   See TLE_DATA_ARCHITECTURE.md for setup instructions."
    echo ""
else
    STARLINK_COUNT=$(ls "$TLE_DIR/starlink/tle"/*.tle 2>/dev/null | wc -l)
    echo "✅ PASSED: TLE data found ($STARLINK_COUNT Starlink files)"
    echo ""
fi

# Summary
echo "========================================================================"
echo "Dependency Check Summary"
echo "========================================================================"
echo ""
echo "✅ orbit-engine: Available"
echo "✅ Algorithm modules: All present"
echo "✅ Python imports: Working"

if [ -d "$TLE_DIR" ]; then
    echo "✅ TLE data: Available"
else
    echo "⚠️  TLE data: Not found (see TLE_DATA_ARCHITECTURE.md)"
fi

echo ""
echo "🎉 handover-rl dependencies verified!"
echo ""
echo "Next steps:"
echo "  1. Install Python dependencies: pip install -r requirements.txt"
echo "  2. Run unit tests: pytest tests/"
echo "  3. Try evaluation: python scripts/evaluate_strategies.py --help"
echo ""
echo "========================================================================"
