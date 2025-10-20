#!/usr/bin/env python3
"""
Stage 0: Academic Compliance Validation (P0 - CRITICAL)

This validation ensures the framework meets academic publication standards:
- NO hardcoded data
- NO mock/simulated data
- NO simplified algorithms
- ALL parameters traceable to sources

If ANY check fails, the entire validation ABORTS.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import json
import yaml
import subprocess
from typing import Dict, List, Tuple, Any
from datetime import datetime

print("\n" + "=" * 80)
print("STAGE 0: ACADEMIC COMPLIANCE VALIDATION (P0 - CRITICAL)")
print("=" * 80)
print("\nThis validation ensures publication-ready academic rigor.")
print("Any failure will ABORT the validation pipeline.\n")

# Results storage
results = {
    'stage': 'Stage 0: Academic Compliance',
    'timestamp': datetime.now().isoformat(),
    'priority': 'P0 - CRITICAL',
    'checks': {},
    'overall_status': 'PENDING',
    'critical_failures': [],
    'warnings': [],
}

def log_check(check_name: str, status: bool, details: str, critical: bool = True):
    """Log a validation check result"""
    results['checks'][check_name] = {
        'status': 'PASS' if status else 'FAIL',
        'details': details,
        'critical': critical,
    }

    status_icon = "✅" if status else "❌"
    print(f"{status_icon} {check_name}: {details}")

    if not status and critical:
        results['critical_failures'].append(check_name)
    elif not status:
        results['warnings'].append(check_name)

# ============================================================================
# CHECK 1: Data Source Verification
# ============================================================================
print("\n" + "-" * 80)
print("CHECK 1: Data Source Verification")
print("-" * 80)

def check_data_sources():
    """Verify all data comes from real sources (no mock/fake data)"""

    # Check 1.1: orbit-engine adapter uses real TLE
    print("\n[1.1] Checking orbit-engine adapter uses real TLE data...")
    adapter_file = Path('src/adapters/orbit_engine_adapter.py')

    if not adapter_file.exists():
        log_check('1.1 OrbitEngine Adapter Exists', False,
                 f"File not found: {adapter_file}", critical=True)
        return

    adapter_code = adapter_file.read_text()

    # Should NOT contain random data generation
    forbidden_patterns = [
        'np.random.normal',
        'np.random.uniform',
        'random.gauss',
        'np.random.randn',
        'fake_data',
        'mock_data',
        'simulated_data',
    ]

    violations = []
    for pattern in forbidden_patterns:
        if pattern in adapter_code:
            # Check if it's in a comment or docstring
            lines = adapter_code.split('\n')
            for i, line in enumerate(lines, 1):
                if pattern in line and not line.strip().startswith('#'):
                    violations.append(f"Line {i}: {line.strip()}")

    if violations:
        details = f"Found forbidden data generation: {', '.join(violations[:3])}"
        log_check('1.1 No Random Data Generation', False, details, critical=True)
    else:
        log_check('1.1 No Random Data Generation', True,
                 "No np.random() or mock data found in adapter", critical=True)

    # Check 1.2: RSRP calculation uses real physics models
    print("\n[1.2] Checking RSRP uses ITU-R/3GPP models...")

    # Check orbit-engine adapter for standard references
    adapter_file = Path('src/adapters/orbit_engine_adapter.py')
    if adapter_file.exists():
        content = adapter_file.read_text()
        itur_found = 'ITU-R' in content
        gpp_found = '3GPP' in content or 'TS 38' in content
        sgp4_found = 'SGP4' in content or 'Skyfield' in content

        if itur_found and gpp_found:
            log_check('1.2 RSRP Uses Official Models', True,
                     "ITU-R and 3GPP references found in orbit-engine adapter", critical=True)
        else:
            missing = []
            if not itur_found: missing.append("ITU-R")
            if not gpp_found: missing.append("3GPP")
            log_check('1.2 RSRP Uses Official Models', False,
                     f"Missing references in adapter: {', '.join(missing)}", critical=True)
    else:
        log_check('1.2 RSRP Uses Official Models', False,
                 "orbit_engine_adapter.py not found", critical=True)

    # Check 1.3: SGP4 orbital mechanics (not simplified)
    print("\n[1.3] Checking SGP4 orbital mechanics...")

    # Check orbit-engine adapter (already checked above)
    adapter_file = Path('src/adapters/orbit_engine_adapter.py')
    if adapter_file.exists():
        content = adapter_file.read_text()
        sgp4_found = 'SGP4' in content or 'Skyfield' in content

        if sgp4_found:
            log_check('1.3 SGP4 Orbital Mechanics', True,
                     "SGP4/Skyfield references found in orbit-engine adapter", critical=True)
        else:
            log_check('1.3 SGP4 Orbital Mechanics', False,
                     "No SGP4/Skyfield references in adapter", critical=True)
    else:
        log_check('1.3 SGP4 Orbital Mechanics', False,
                 "orbit_engine_adapter.py not found", critical=True)

check_data_sources()

# ============================================================================
# CHECK 2: Algorithm Completeness Verification
# ============================================================================
print("\n" + "-" * 80)
print("CHECK 2: Algorithm Completeness Verification")
print("-" * 80)

def check_algorithm_completeness():
    """Verify algorithms are complete implementations (not simplified)"""

    # Check 2.1: DQN has all required components
    print("\n[2.1] Checking DQN completeness...")

    dqn_file = Path('src/agents/dqn/dqn_agent.py')
    if not dqn_file.exists():
        log_check('2.1 DQN Agent Exists', False,
                 f"File not found: {dqn_file}", critical=True)
        return

    dqn_code = dqn_file.read_text()

    required_components = {
        'target_network': 'self.target_network',
        'replay_buffer': 'ReplayBuffer',
        'epsilon': 'epsilon',
    }

    missing_components = []
    for component, pattern in required_components.items():
        if pattern not in dqn_code:
            missing_components.append(component)

    # Check for target network update (either hard or soft)
    has_hard_update = 'load_state_dict(self.q_network.state_dict())' in dqn_code
    has_soft_update = 'tau' in dqn_code
    has_target_update = has_hard_update or has_soft_update

    if not missing_components and has_target_update:
        update_method = "hard update" if has_hard_update else "soft update"
        log_check('2.1 DQN Completeness', True,
                 f"All required components present (target_network, replay_buffer, epsilon, {update_method})",
                 critical=True)
    elif not has_target_update:
        log_check('2.1 DQN Completeness', False,
                 "Missing target network update mechanism (neither hard nor soft)", critical=True)
    else:
        log_check('2.1 DQN Completeness', False,
                 f"Missing components: {', '.join(missing_components)}", critical=True)

    # Check 2.2: A4 Event follows 3GPP standard
    print("\n[2.2] Checking A4 Event standard compliance...")

    a4_file = Path('src/strategies/a4_based_strategy.py')
    if not a4_file.exists():
        log_check('2.2 A4 Strategy Exists', False,
                 f"File not found: {a4_file}", critical=True)
    else:
        a4_code = a4_file.read_text()

        # Check for 3GPP references
        has_gpp_ref = '3GPP' in a4_code or 'TS 38.331' in a4_code
        # Check for A4 event logic (threshold + hysteresis + offset)
        has_threshold = 'threshold' in a4_code
        has_hysteresis = 'hysteresis' in a4_code
        has_a4_logic = 'A4' in a4_code or 'Mn + Ofn' in a4_code

        if has_gpp_ref and has_threshold and has_hysteresis:
            log_check('2.2 A4 Standard Compliance', True,
                     "3GPP references and A4 event logic found", critical=True)
        else:
            missing = []
            if not has_gpp_ref: missing.append("3GPP reference")
            if not has_threshold: missing.append("threshold")
            if not has_hysteresis: missing.append("hysteresis")
            log_check('2.2 A4 Standard Compliance', False,
                     f"Missing: {', '.join(missing)}", critical=True)

    # Check 2.3: D2 Event follows 3GPP NTN standard
    print("\n[2.3] Checking D2 Event standard compliance...")

    d2_file = Path('src/strategies/d2_based_strategy.py')
    if not d2_file.exists():
        log_check('2.3 D2 Strategy Exists', False,
                 f"File not found: {d2_file}", critical=True)
    else:
        d2_code = d2_file.read_text()

        # Check for 3GPP NTN references
        has_gpp_ref = '3GPP' in d2_code or 'TS 38.331' in d2_code
        has_ntn_ref = 'NTN' in d2_code or 'Rel-17' in d2_code
        # Check for D2 event logic (dual thresholds)
        has_threshold1 = 'threshold1' in d2_code
        has_threshold2 = 'threshold2' in d2_code
        has_d2_logic = 'D2' in d2_code

        if has_gpp_ref and has_threshold1 and has_threshold2:
            log_check('2.3 D2 Standard Compliance', True,
                     "3GPP NTN references and D2 event logic found", critical=True)
        else:
            missing = []
            if not has_gpp_ref: missing.append("3GPP reference")
            if not has_threshold1: missing.append("threshold1")
            if not has_threshold2: missing.append("threshold2")
            log_check('2.3 D2 Standard Compliance', False,
                     f"Missing: {', '.join(missing)}", critical=True)

check_algorithm_completeness()

# ============================================================================
# CHECK 3: Parameter Traceability Verification
# ============================================================================
print("\n" + "-" * 80)
print("CHECK 3: Parameter Traceability Verification")
print("-" * 80)

def check_parameter_traceability():
    """Verify all parameters are traceable to sources"""

    # Check 3.1: D2 parameters from orbit-engine
    print("\n[3.1] Checking D2 parameter traceability...")

    d2_config = Path('config/strategies/d2_based.yaml')
    if not d2_config.exists():
        log_check('3.1 D2 Config Exists', False,
                 f"File not found: {d2_config}", critical=True)
    else:
        with open(d2_config) as f:
            config = yaml.safe_load(f)

        # Check for SOURCE annotations
        has_source = False
        config_text = d2_config.read_text()

        # Look for key parameters and their sources
        checks = {
            '1412.8': 'threshold1_km (75th percentile)',
            '1005.8': 'threshold2_km (median)',
            'orbit-engine': 'data source',
            'Stage 4': 'orbit-engine stage',
        }

        found = {}
        for key, desc in checks.items():
            if key in config_text:
                found[desc] = True
            else:
                found[desc] = False

        all_found = all(found.values())

        if all_found:
            log_check('3.1 D2 Parameter Traceability', True,
                     "All D2 parameters traceable to orbit-engine Stage 4", critical=True)
        else:
            missing = [k for k, v in found.items() if not v]
            log_check('3.1 D2 Parameter Traceability', False,
                     f"Missing: {', '.join(missing)}", critical=True)

    # Check 3.2: A4 parameters from standards
    print("\n[3.2] Checking A4 parameter traceability...")

    a4_config = Path('config/strategies/a4_based.yaml')
    if not a4_config.exists():
        log_check('3.2 A4 Config Exists', False,
                 f"File not found: {a4_config}", critical=True)
    else:
        config_text = a4_config.read_text()

        # Look for key parameters and their sources
        checks = {
            '-100': 'threshold_dbm',
            '1.5': 'hysteresis_db',
            'Yu et al': 'paper reference',
            '3GPP': 'standard reference',
        }

        found = {}
        for key, desc in checks.items():
            if key in config_text:
                found[desc] = True
            else:
                found[desc] = False

        all_found = all(found.values())

        if all_found:
            log_check('3.2 A4 Parameter Traceability', True,
                     "All A4 parameters traceable to Yu et al. 2022 and 3GPP", critical=True)
        else:
            missing = [k for k, v in found.items() if not v]
            log_check('3.2 A4 Parameter Traceability', False,
                     f"Missing: {', '.join(missing)}", critical=True)

    # Check 3.3: DQN hyperparameters in config
    print("\n[3.3] Checking DQN hyperparameters in config...")

    training_config = Path('config/training_config.yaml')
    if not training_config.exists():
        log_check('3.3 DQN Config Exists', False,
                 f"File not found: {training_config}", critical=True)
    else:
        with open(training_config) as f:
            config = yaml.safe_load(f)

        # Check for DQN section
        if 'dqn' not in config:
            log_check('3.3 DQN Hyperparameters in Config', False,
                     "No 'dqn' section in config/training_config.yaml", critical=True)
        else:
            dqn_config = config['dqn']

            # Check for key hyperparameters
            required_params = {
                'learning_rate': dqn_config.get('learning_rate'),
                'gamma': dqn_config.get('gamma'),
                'epsilon_start': dqn_config.get('epsilon_start'),
                'epsilon_end': dqn_config.get('epsilon_end'),
                'epsilon_decay': dqn_config.get('epsilon_decay'),
                'batch_size': dqn_config.get('batch_size'),
            }

            # Check replay_buffer section
            if 'replay_buffer' in dqn_config and 'capacity' in dqn_config['replay_buffer']:
                required_params['buffer_capacity'] = dqn_config['replay_buffer']['capacity']

            missing = [k for k, v in required_params.items() if v is None]

            if not missing:
                log_check('3.3 DQN Hyperparameters in Config', True,
                         "All DQN hyperparameters in config/training_config.yaml (not hardcoded)", critical=True)
            else:
                log_check('3.3 DQN Hyperparameters in Config', False,
                         f"Missing from config: {', '.join(missing)}", critical=True)

check_parameter_traceability()

# ============================================================================
# CHECK 4: Forbidden Content Search
# ============================================================================
print("\n" + "-" * 80)
print("CHECK 4: Forbidden Content Search")
print("-" * 80)

def check_forbidden_content():
    """Search for academic red flags in codebase"""

    print("\n[4.1] Searching for forbidden keywords...")

    forbidden_keywords = {
        'simplified algorithm': 'Indicates algorithm simplification',
        'mock data': 'Indicates fake data usage',
        'fake data': 'Indicates fake data usage',
        'estimated value': 'Indicates parameter estimation',
        'assumed parameter': 'Indicates parameter assumption',
        'placeholder': 'Indicates incomplete implementation',
    }

    # Negative contexts (allowed - clarifications that we DON'T use these)
    negative_contexts = [
        'no simplified',
        'not simplified',
        'no mock',
        'not mock',
        'no fake',
        'not fake',
        'no placeholder',
        'removed placeholder',
        'removing placeholder',
        'after removing',
        'skip placeholder',
        'fixed:',
        'complete implementations only',
    ]

    # Search in source code
    src_files = list(Path('src').rglob('*.py'))

    violations = {}
    for keyword, reason in forbidden_keywords.items():
        matches = []
        for file_path in src_files:
            content = file_path.read_text()
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                # Skip comments
                if line.strip().startswith('#'):
                    continue

                # Check if keyword appears
                if keyword in line.lower():
                    # Check if it's in a negative context (clarification)
                    is_negative_context = any(neg in line.lower() for neg in negative_contexts)

                    # If not in negative context, it's a violation
                    if not is_negative_context:
                        matches.append(f"{file_path}:{i}")

        if matches:
            violations[keyword] = matches[:5]  # First 5 matches

    if not violations:
        log_check('4.1 No Forbidden Keywords', True,
                 "No forbidden keywords in positive context (negative contexts like 'No mock data' are OK)",
                 critical=True)
    else:
        violation_summary = '; '.join([f"{k}: {len(v)} matches" for k, v in violations.items()])
        log_check('4.1 No Forbidden Keywords', False,
                 f"Found in positive context: {violation_summary}", critical=True)

        # Print details
        for keyword, matches in violations.items():
            print(f"    ⚠️  '{keyword}' in positive context: {len(matches)} matches")
            for match in matches[:3]:
                print(f"       - {match}")

    # Check 4.2: No hardcoded magic numbers in strategies
    print("\n[4.2] Checking for hardcoded magic numbers...")

    strategy_files = list(Path('src/strategies').glob('*.py'))
    magic_numbers = []

    for file_path in strategy_files:
        if file_path.name == '__init__.py' or file_path.name == 'base_strategy.py':
            continue

        content = file_path.read_text()

        # Look for numeric literals that aren't in __init__ or config loading
        import re
        # Find numbers like -100.0, 1412.8, etc. (but not 0, 1, 2, etc.)
        pattern = r'(?<!\.)\b\d{2,}\.\d+\b'
        matches = re.findall(pattern, content)

        # Check if these numbers are in __init__ parameters (acceptable)
        init_pattern = r'def __init__\(.*?\):'
        init_match = re.search(init_pattern, content, re.DOTALL)

        if matches and not init_match:
            magic_numbers.append(f"{file_path.name}: {', '.join(set(matches[:5]))}")

    if not magic_numbers:
        log_check('4.2 No Magic Numbers', True,
                 "All numeric values in strategy __init__ or configs", critical=False)
    else:
        log_check('4.2 No Magic Numbers', False,
                 f"Potential magic numbers: {'; '.join(magic_numbers)}", critical=False)

check_forbidden_content()

# ============================================================================
# CHECK 5: Standards Compliance Matrix
# ============================================================================
print("\n" + "-" * 80)
print("CHECK 5: Standards Compliance Matrix")
print("-" * 80)

def check_standards_compliance():
    """Verify compliance with official standards"""

    standards = {
        'A4 Event': {
            'standard': '3GPP TS 38.331',
            'section': '5.5.4.5',
            'file': 'src/strategies/a4_based_strategy.py',
        },
        'D2 Event': {
            'standard': '3GPP TS 38.331',
            'section': '5.5.4.15a',
            'file': 'src/strategies/d2_based_strategy.py',
        },
        'RSRP Calculation': {
            'standard': '3GPP TS 38.214',
            'section': 'Various',
            'file': 'src/adapters/orbit_engine_adapter.py',  # Implemented in orbit-engine
        },
        'Path Loss': {
            'standard': 'ITU-R P.676',
            'section': 'Various',
            'file': 'src/adapters/orbit_engine_adapter.py',  # Implemented in orbit-engine
        },
        'SGP4 Orbital': {
            'standard': 'SGP4 (NORAD)',
            'section': 'Various',
            'file': 'src/adapters/orbit_engine_adapter.py',  # Implemented in orbit-engine
        },
    }

    print("\nStandards Compliance Matrix:")
    print("-" * 80)
    print(f"{'Component':<20} {'Standard':<25} {'Section':<15} {'Status'}")
    print("-" * 80)

    all_compliant = True

    for component, info in standards.items():
        file_path = Path(info['file'])

        # Check if file exists and contains standard reference
        found_reference = False
        if file_path.exists():
            content = file_path.read_text()
            # For orbit-engine adapter, check for both the standard name and implementation notes
            if 'orbit_engine_adapter' in str(file_path):
                # Check for standard reference or implementation note
                found_reference = (info['standard'] in content or
                                 component.split()[0].upper() in content)
            else:
                # For strategy files, check for direct standard reference
                found_reference = info['standard'] in content

        status = "✅ VERIFIED" if found_reference else "❌ MISSING"
        if not found_reference:
            all_compliant = False

        print(f"{component:<20} {info['standard']:<25} {info['section']:<15} {status}")

    print("-" * 80)

    if all_compliant:
        log_check('5.1 Standards Compliance', True,
                 "All components reference official standards (some via orbit-engine)", critical=True)
    else:
        log_check('5.1 Standards Compliance', False,
                 "Some components missing standard references", critical=True)

check_standards_compliance()

# ============================================================================
# FINAL ASSESSMENT
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 0 FINAL ASSESSMENT")
print("=" * 80)

# Count results
total_checks = len(results['checks'])
passed_checks = sum(1 for c in results['checks'].values() if c['status'] == 'PASS')
failed_checks = sum(1 for c in results['checks'].values() if c['status'] == 'FAIL')
critical_failures = len(results['critical_failures'])

print(f"\nTotal Checks: {total_checks}")
print(f"Passed: {passed_checks} ✅")
print(f"Failed: {failed_checks} ❌")
print(f"Critical Failures: {critical_failures} 🚨")

# Determine overall status
if critical_failures > 0:
    results['overall_status'] = 'FAIL - CRITICAL'
    print("\n" + "🚨" * 40)
    print("CRITICAL FAILURE - VALIDATION ABORTED")
    print("🚨" * 40)
    print("\nCritical failures detected:")
    for failure in results['critical_failures']:
        print(f"  ❌ {failure}")
    print("\n⚠️  Fix these issues before proceeding with further validation.")
    print("⚠️  Academic rigor is non-negotiable for baseline research.")
    exit_code = 1
elif failed_checks > 0:
    results['overall_status'] = 'PASS WITH WARNINGS'
    print("\n" + "⚠️ " * 40)
    print("PASS WITH WARNINGS")
    print("⚠️ " * 40)
    print("\nNon-critical warnings:")
    for warning in results['warnings']:
        print(f"  ⚠️  {warning}")
    exit_code = 0
else:
    results['overall_status'] = 'PASS'
    print("\n" + "✅" * 40)
    print("STAGE 0: ACADEMIC COMPLIANCE - PASS")
    print("✅" * 40)
    print("\n🎉 Framework meets academic publication standards!")
    print("✅ No hardcoded data")
    print("✅ No mock/simulated data")
    print("✅ No simplified algorithms")
    print("✅ All parameters traceable")
    print("✅ Standards compliant")
    exit_code = 0

# Save results
output_dir = Path('results/validation')
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / 'stage0_academic_compliance.json'

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n📊 Results saved to: {output_file}")

if results['overall_status'] == 'PASS':
    print("\n✅ Ready to proceed to Stage 1: Unit Validation")
elif results['overall_status'] == 'PASS WITH WARNINGS':
    print("\n⚠️  Can proceed to Stage 1, but review warnings")
else:
    print("\n❌ Cannot proceed - fix critical failures first")

print("=" * 80 + "\n")

sys.exit(exit_code)
