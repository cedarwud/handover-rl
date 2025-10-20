#!/bin/bash
# ========================================
# Handover-RL Environment Setup Script
# ========================================
# Usage: ./setup_env.sh
#
# This script:
#   1. Validates Python version
#   2. Creates virtual environment
#   3. Installs dependencies
#   4. Verifies orbit-engine integration
#   5. Creates .env file

set -e  # Exit on error

echo "========================================="
echo "🚀 Handover-RL Environment Setup"
echo "========================================="
echo ""

# ============================================================================
# Step 1: Check Python version
# ============================================================================
echo "📋 Step 1/6: Checking Python version..."

PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "❌ ERROR: python3 not found"
    echo "   Please install Python 3.12+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

echo "   Found: Python $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MINOR" -lt 10 ]; then
    echo "❌ ERROR: Python 3.10+ required, found $PYTHON_VERSION"
    exit 1
fi

if [ "$PYTHON_MINOR" -lt 12 ]; then
    echo "⚠️  WARNING: Python 3.12+ recommended for best compatibility"
    echo "   Current: Python $PYTHON_VERSION"
    read -p "   Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "✅ Python version OK"
echo ""

# ============================================================================
# Step 2: Check orbit-engine existence
# ============================================================================
echo "📋 Step 2/6: Checking orbit-engine integration..."

ORBIT_ENGINE_PATH="../orbit-engine"
if [ ! -d "$ORBIT_ENGINE_PATH" ]; then
    echo "❌ ERROR: orbit-engine not found at $ORBIT_ENGINE_PATH"
    echo ""
    echo "Please ensure directory structure:"
    echo "  /path/to/satellite/"
    echo "    ├── orbit-engine/     ← Required"
    echo "    └── handover-rl/      ← You are here"
    echo ""
    echo "Clone orbit-engine:"
    echo "  cd .."
    echo "  git clone https://github.com/user/orbit-engine.git"
    exit 1
fi

echo "✅ orbit-engine found at $ORBIT_ENGINE_PATH"
echo ""

# ============================================================================
# Step 3: Create virtual environment
# ============================================================================
echo "📋 Step 3/6: Creating virtual environment..."

if [ -d "venv" ]; then
    echo "⚠️  venv directory already exists"
    read -p "   Remove and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Removing old venv..."
        rm -rf venv
    else
        echo "   Skipping venv creation"
    fi
fi

if [ ! -d "venv" ]; then
    echo "   Creating venv with Python $PYTHON_VERSION..."
    $PYTHON_CMD -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Using existing virtual environment"
fi

echo ""

# ============================================================================
# Step 4: Activate environment and upgrade pip
# ============================================================================
echo "📋 Step 4/6: Activating environment and upgrading pip..."

source venv/bin/activate

echo "   Upgrading pip, setuptools, wheel..."
python -m pip install --upgrade pip setuptools wheel > /dev/null

echo "✅ pip upgraded"
echo ""

# ============================================================================
# Step 5: Install dependencies
# ============================================================================
echo "📋 Step 5/6: Installing dependencies..."
echo ""

echo "   📥 Installing all dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "✅ All dependencies installed"
echo ""

# ============================================================================
# Step 6: Verify orbit-engine integration
# ============================================================================
echo "📋 Step 6/6: Verifying orbit-engine integration..."

python3 -c "
import sys
sys.path.insert(0, '$ORBIT_ENGINE_PATH')

# Test imports
try:
    from src.stages.stage5_signal_analysis.gpp_ts38214_signal_calculator import create_3gpp_signal_calculator
    from src.stages.stage5_signal_analysis.itur_physics_calculator import create_itur_physics_calculator
    from src.stages.stage5_signal_analysis.itur_official_atmospheric_model import create_itur_official_model
    print('✅ All orbit-engine imports successful')
except ImportError as e:
    print(f'❌ orbit-engine import failed: {e}')
    sys.exit(1)

# Test factory functions
try:
    config = {
        'bandwidth_mhz': 100,
        'subcarrier_spacing_khz': 30,
        'noise_figure_db': 3.0,
        'temperature_k': 290.0
    }
    calc = create_3gpp_signal_calculator(config)
    print(f'✅ 3GPP calculator created: N_RB={calc.n_rb}')
except Exception as e:
    print(f'❌ Factory function test failed: {e}')
    sys.exit(1)
" || {
    echo ""
    echo "❌ orbit-engine integration failed"
    echo ""
    echo "Possible causes:"
    echo "  1. orbit-engine not properly installed"
    echo "  2. orbit-engine missing dependencies"
    echo ""
    echo "Try:"
    echo "  cd $ORBIT_ENGINE_PATH"
    echo "  ./setup.sh"
    echo "  source venv/bin/activate"
    echo "  ./run.sh --stage 5  # Verify orbit-engine works"
    exit 1
}

echo ""

# ============================================================================
# Step 7: Create .env file if not exists
# ============================================================================
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << 'ENVEOF'
# Handover-RL Environment Configuration
# Generated by setup_env.sh

# Orbit-Engine Path (relative to handover-rl)
ORBIT_ENGINE_PATH=../orbit-engine

# Python Path (automatically includes orbit-engine)
PYTHONPATH=${ORBIT_ENGINE_PATH}:${PYTHONPATH}

# Logging
LOG_LEVEL=INFO

# Development
DEBUG=0
ENVEOF
    echo "✅ .env file created"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "========================================="
echo "✅ Environment setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Verify adapter integration:"
echo "   python -c 'from adapters.orbit_engine_adapter import OrbitEngineAdapter; print(\"✅ Ready\")'"
echo ""
echo "3. Generate training data:"
echo "   python scripts/generate_dataset.py --episodes 10"
echo ""
echo "4. Train DQN agent:"
echo "   python scripts/train_dqn.py --config config/training_config.yaml"
echo ""
