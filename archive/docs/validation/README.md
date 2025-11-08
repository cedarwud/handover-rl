# Validation Framework for Refactored RL Framework

## Quick Start

### 1. Quick Academic Compliance Check (5 minutes)
```bash
python scripts/validation/stage0_academic_compliance.py
```

**Use this** before starting any experiments to ensure academic rigor.

### 2. Unit Tests (10 minutes)
```bash
python scripts/validation/stage1_unit_tests.py
```

**Use this** after code changes to verify basic functionality.

### 3. Full Validation (4-5 hours)
```bash
./scripts/validation/run_full_validation.sh
```

**Use this**:
- Before submitting papers
- Before running baseline experiments
- After major refactoring

---

## Validation Stages

| Stage | Name | Duration | Priority | Purpose |
|-------|------|----------|----------|---------|
| 0 | Academic Compliance | ~5 min | **P0 CRITICAL** | Verify no mock data, no simplified algorithms |
| 1 | Unit Tests | ~10 min | P1 | Test individual components |
| 2 | Integration Tests | ~30 min | P1 | Test component interactions |
| 3 | E2E Baseline (Level 1) | ~2 hours | P1 | Test rule-based baselines |
| 4 | E2E DQN (Level 1) | ~2 hours | P1 | Test DQN training |

---

## Output Files

After running validation:

```
results/validation/
├── stage0_academic_compliance.json    ← P0 critical results
├── stage1_unit_tests.json
├── stage3_baseline_comparison.csv     ← Baseline performance
└── stage4_dqn/
    ├── metrics.csv                    ← DQN training metrics
    ├── checkpoints/final_model.pt
    └── tensorboard/

docs/validation/
└── VALIDATION_REPORT.md               ← Auto-generated report
```

---

## Success Criteria

### Stage 0 (MANDATORY 100% PASS)
- ✅ No hardcoded data
- ✅ No mock/simulated data
- ✅ No simplified algorithms
- ✅ All parameters traceable
- ✅ Standards compliant (3GPP, ITU-R)

**If Stage 0 fails → Fix immediately, cannot proceed**

### Stage 1-4
- ✅ All tests pass
- ✅ Baselines work correctly
- ✅ DQN training produces learning curve

---

## When to Run

### Always (Stage 0 only - 5 min)
- Before any experiments
- After changing algorithms
- Before paper submission

### After Code Changes (Stage 0-1 - 15 min)
- After implementing new features
- After bug fixes
- After refactoring

### Full Validation (All stages - 4-5 hours)
- Before baseline experiments
- Before paper submission
- After major refactoring
- Monthly (recommended)

---

## Troubleshooting

### Stage 0 Fails
**Problem**: Found "simplified" keyword in code
**Solution**: Search and remove all simplified algorithms

**Problem**: Missing standard references (3GPP, ITU-R)
**Solution**: Add proper citations in docstrings

**Problem**: Parameters not traceable
**Solution**: Add SOURCE annotations to config files

### Stage 1 Fails
**Problem**: Protocol compliance fails
**Solution**: Ensure all agents/strategies implement required methods

### Stage 3/4 Timeout
**Problem**: Level 1 taking too long
**Solution**: Use Level 0 for quick testing instead

---

## Quick Reference

```bash
# P0 Check only (5 min)
python scripts/validation/stage0_academic_compliance.py

# Unit tests only (10 min)
python scripts/validation/stage1_unit_tests.py

# Full validation (4-5 hours)
./scripts/validation/run_full_validation.sh

# Demo comparison (10 min - Level 0)
python scripts/demo_comparison.py
```

---

## Documentation

- **VALIDATION_PLAN.md** - Detailed validation plan
- **VALIDATION_REPORT.md** - Generated after running validation
- **README.md** - This file

---

**Created**: 2025-10-20
**Version**: 1.0
**Status**: Ready for use
