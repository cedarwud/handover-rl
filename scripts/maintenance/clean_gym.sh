#!/bin/bash
# Clean up old gym and use only gymnasium

set -e

echo "🧹 Cleaning up old gym (OpenAI) and keeping only gymnasium (Farama)"
echo ""

# Check if in venv
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: Not in virtual environment!"
    echo "Activating venv..."
    source venv/bin/activate
fi

echo "Current gym packages:"
pip list | grep gym

echo ""
echo "Uninstalling old gym..."
pip uninstall gym gym-notices -y

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Remaining packages:"
pip list | grep gym

echo ""
echo "Verifying gymnasium works:"
python3 << 'PYEOF'
import gymnasium
print(f"✅ Gymnasium {gymnasium.__version__} is working!")
print(f"✅ Location: {gymnasium.__file__}")
PYEOF

echo ""
echo "🎉 Done! Now only using Gymnasium (Farama Foundation)"
