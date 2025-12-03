#!/usr/bin/env python3
"""
Analyze multi-seed results for academic reporting
Following JMLR 2024 standards for RL evaluation
"""

import re
import numpy as np
from pathlib import Path

def extract_metrics_from_log(log_file):
    """Extract handover and reward metrics from evaluation log"""
    with open(log_file, 'r') as f:
        content = f.read()

    # Extract mean, std from handover section (first occurrence before "Handover Distribution")
    # Pattern: "   Mean:    0.94" followed by "   Std:     1.08"
    handover_section = content.split('📈 Handover Distribution:')[0]

    # Find the first Mean and Std that appear close together
    lines = handover_section.split('\n')
    mean = None
    std = None

    for i, line in enumerate(lines):
        if 'Mean:' in line and mean is None:
            # Extract number from line like "   Mean:    0.94"
            mean_match = re.search(r'Mean:\s+([\d.]+)', line)
            if mean_match:
                mean = float(mean_match.group(1))
                # Look for Std in next few lines
                for j in range(i+1, min(i+5, len(lines))):
                    if 'Std:' in lines[j]:
                        std_match = re.search(r'Std:\s+([\d.]+)', lines[j])
                        if std_match:
                            std = float(std_match.group(1))
                            break
                if std is not None:
                    break

    if mean is not None and std is not None:
        return mean, std
    return None, None

def main():
    seeds = [42, 123, 456, 789, 2024]
    handover_means = []
    handover_stds = []

    print("=" * 80)
    print("DQN BASELINE - MULTI-SEED ANALYSIS (5 SEEDS)")
    print("=" * 80)
    print()

    # Collect metrics from each seed
    for seed in seeds:
        log_file = f"/tmp/eval_seed{seed}.log"
        mean, std = extract_metrics_from_log(log_file)

        if mean is not None:
            handover_means.append(mean)
            handover_stds.append(std)
            print(f"Seed {seed:4d}: {mean:.2f} ± {std:.2f} handovers")
        else:
            print(f"Seed {seed:4d}: ⚠️  Log file not found or incomplete")

    if len(handover_means) == 0:
        print("\\n❌ No valid results found. Check evaluation logs.")
        return

    print()
    print("=" * 80)
    print("AGGREGATED STATISTICS (Across Seeds)")
    print("=" * 80)

    # Calculate across-seed statistics
    mean_of_means = np.mean(handover_means)
    std_of_means = np.std(handover_means, ddof=1)  # Sample std
    sem_of_means = std_of_means / np.sqrt(len(handover_means))

    # Calculate IQM (Interquartile Mean) - JMLR 2024 recommendation
    q25 = np.percentile(handover_means, 25)
    q75 = np.percentile(handover_means, 75)
    iqm_values = [x for x in handover_means if q25 <= x <= q75]
    iqm = np.mean(iqm_values) if iqm_values else mean_of_means

    # 95% Confidence Interval
    ci_lower = mean_of_means - 1.96 * sem_of_means
    ci_upper = mean_of_means + 1.96 * sem_of_means

    print(f"\\n📊 Mean Handovers (Mean ± SD):")
    print(f"   {mean_of_means:.2f} ± {std_of_means:.2f}")
    print(f"\\n📊 Standard Error of Mean:")
    print(f"   {sem_of_means:.3f}")
    print(f"\\n📊 95% Confidence Interval:")
    print(f"   [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"\\n📊 Interquartile Mean (IQM):")
    print(f"   {iqm:.2f}")
    print(f"\\n📊 Min/Max across seeds:")
    print(f"   [{min(handover_means):.2f}, {max(handover_means):.2f}]")

    print()
    print("=" * 80)
    print("ACADEMIC REPORTING FORMAT")
    print("=" * 80)
    print()
    print(f"DQN Baseline (SB3, {len(handover_means)} seeds, 2500 training episodes):")
    print(f"  Mean handovers: {mean_of_means:.2f} ± {std_of_means:.2f}")
    print(f"  95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"  IQM: {iqm:.2f}")
    print(f"  Evaluation: 100 episodes per seed (total {len(handover_means) * 100} episodes)")
    print()

    # Statistical significance check
    cv = std_of_means / mean_of_means  # Coefficient of variation
    if cv < 0.1:
        print("✅ Low variability across seeds (<10% CV)")
    elif cv < 0.2:
        print("⚠️  Moderate variability across seeds (10-20% CV)")
    else:
        print("⚠️  High variability across seeds (>20% CV) - consider more seeds")

    print()
    print("=" * 80)

    # Save to file
    output_file = "output/multi_seed_results.txt"
    Path("output").mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        f.write(f"DQN Baseline Multi-Seed Results\\n")
        f.write(f"================================\\n\\n")
        f.write(f"Number of seeds: {len(handover_means)}\\n")
        f.write(f"Training episodes per seed: 2500\\n")
        f.write(f"Evaluation episodes per seed: 100\\n\\n")
        f.write(f"Results:\\n")
        for seed, mean, std in zip(seeds[:len(handover_means)], handover_means, handover_stds):
            f.write(f"  Seed {seed}: {mean:.2f} ± {std:.2f}\\n")
        f.write(f"\\nAggregated Statistics:\\n")
        f.write(f"  Mean: {mean_of_means:.2f} ± {std_of_means:.2f}\\n")
        f.write(f"  SEM: {sem_of_means:.3f}\\n")
        f.write(f"  95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]\\n")
        f.write(f"  IQM: {iqm:.2f}\\n")
        f.write(f"  CV: {cv:.2%}\\n")

    print(f"Results saved to: {output_file}")

if __name__ == '__main__':
    main()
