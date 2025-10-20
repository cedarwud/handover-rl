#!/usr/bin/env python3
"""
Phase 0 - Step 0.2: Verify Dependencies

Check all required packages for the refactor are installed
"""

import sys
from importlib import import_module
from pathlib import Path

def check_package(package_name, import_name=None, min_version=None):
    """Check if a package is installed and optionally verify version"""
    if import_name is None:
        import_name = package_name

    try:
        module = import_module(import_name)
        version = getattr(module, '__version__', 'unknown')

        if min_version and version != 'unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                print(f"⚠️  {package_name:20s} = {version:10s} (< {min_version} required)")
                return False

        print(f"✅ {package_name:20s} = {version:10s}")
        return True

    except ImportError:
        print(f"❌ {package_name:20s} = NOT INSTALLED")
        return False

def main():
    print("=" * 80)
    print("Phase 0 - Step 0.2: Verify Dependencies")
    print("=" * 80)

    print("\n[1/3] Checking core scientific packages...")
    print("-" * 80)

    core_packages = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('astropy', 'astropy'),
        ('skyfield', 'skyfield'),
    ]

    core_ok = True
    for pkg, import_name in core_packages:
        if not check_package(pkg, import_name):
            core_ok = False

    print("\n[2/3] Checking reinforcement learning packages...")
    print("-" * 80)

    rl_packages = [
        ('gymnasium', 'gymnasium'),  # OpenAI Gym replacement
        ('torch', 'torch'),  # PyTorch for DQN
        ('tensorboard', 'tensorboard'),  # For logging
    ]

    rl_ok = True
    for pkg, import_name in rl_packages:
        if not check_package(pkg, import_name):
            rl_ok = False

    print("\n[3/3] Checking utility packages...")
    print("-" * 80)

    util_packages = [
        ('yaml', 'yaml'),  # PyYAML
        ('tqdm', 'tqdm'),  # Progress bars
        ('requests', 'requests'),  # HTTP requests
    ]

    util_ok = True
    for pkg, import_name in util_packages:
        if not check_package(pkg, import_name):
            util_ok = False

    # Check project structure
    print("\n[4/3] Checking project structure...")
    print("-" * 80)

    required_dirs = [
        'src',
        'src/adapters',
        'src/environments',
        'src/agents',
        'config',
        'refactor_plan',
    ]

    structure_ok = True
    for dir_path in required_dirs:
        full_path = Path(dir_path)
        if full_path.exists():
            print(f"✅ Directory: {dir_path}")
        else:
            print(f"❌ Directory: {dir_path} (missing - will create)")
            structure_ok = False

    # Summary
    print("\n" + "=" * 80)
    print("📊 Dependency Check Summary")
    print("=" * 80)

    all_ok = core_ok and rl_ok and util_ok and structure_ok

    if core_ok:
        print("✅ Core scientific packages: OK")
    else:
        print("❌ Core scientific packages: MISSING PACKAGES")
        print("   Fix: pip install numpy scipy pandas matplotlib astropy skyfield")

    if rl_ok:
        print("✅ Reinforcement learning packages: OK")
    else:
        print("❌ Reinforcement learning packages: MISSING PACKAGES")
        print("   Fix: pip install gymnasium torch tensorboard")

    if util_ok:
        print("✅ Utility packages: OK")
    else:
        print("❌ Utility packages: MISSING PACKAGES")
        print("   Fix: pip install pyyaml tqdm requests")

    if not structure_ok:
        print("⚠️  Project structure: INCOMPLETE (will create missing directories)")
        print("\n📁 Creating missing directories...")
        for dir_path in required_dirs:
            full_path = Path(dir_path)
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"   Created: {dir_path}")
    else:
        print("✅ Project structure: OK")

    print("\n" + "=" * 80)

    if all_ok:
        print("✅ VERIFICATION PASSED - All dependencies ready")
        print("   Ready to proceed to Phase 0: Step 0.3 (Create test framework)")
        return True
    else:
        print("❌ VERIFICATION FAILED - Missing dependencies")
        print("\n   To install all missing packages:")
        print("   pip install numpy scipy pandas matplotlib astropy skyfield \\")
        print("               gymnasium torch tensorboard pyyaml tqdm requests")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
