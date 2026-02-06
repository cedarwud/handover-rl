#!/usr/bin/env python3
"""
Comprehensive DQN Evaluation

Evaluates trained DQN models on full metrics suite:
- Episode rewards
- Handover frequency
- Connectivity percentage
- RSRP statistics
- QoS metrics

Compares against baseline policies to validate whether DQN
optimizes for objectives beyond raw reward.
"""

import sys
sys.path.insert(0, 'src')

import argparse
import yaml
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from stable_baselines3 import DQN
import warnings
import logging

from utils.satellite_utils import load_stage4_optimized_satellites
from environments import SatelliteHandoverEnv
from adapters.adapter_wrapper import AdapterWrapper

warnings.filterwarnings('ignore')
logging.disable(logging.INFO)


def evaluate_dqn_model(model_path: str, env, num_episodes: int = 100, seed: int = None):
    """Evaluate a single DQN model comprehensively

    Args:
        model_path: Path to trained DQN model
        env: Environment instance
        num_episodes: Number of episodes to evaluate
        seed: Random seed for this model

    Returns:
        dict: Comprehensive statistics
    """

    print(f"\n{'='*80}")
    print(f"Evaluating DQN model: {model_path}")
    if seed:
        print(f"Seed: {seed}")
    print(f"{'='*80}\n")

    # Load model
    model = DQN.load(model_path)

    # Metrics storage
    episode_rewards = []
    episode_handovers = []
    episode_lengths = []
    episode_avg_rsrp = []
    episode_connectivity = []

    # Run episodes
    for ep in tqdm(range(num_episodes), desc=f"Seed {seed if seed else 'Unknown'}"):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0
        rsrp_sum = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            ep_length += 1

            # Collect RSRP (current satellite is first in observation)
            if len(obs) > 0:
                rsrp_sum += obs[0, 10]  # RSRP at index 10

        # Store episode metrics
        episode_rewards.append(ep_reward)
        episode_handovers.append(env.episode_stats['num_handovers'])
        episode_lengths.append(ep_length)
        episode_avg_rsrp.append(rsrp_sum / ep_length if ep_length > 0 else 0)
        episode_connectivity.append(
            env.episode_stats['connected_steps'] / ep_length if ep_length > 0 else 0
        )

    # Calculate statistics
    results = {
        'seed': seed,
        'num_episodes': num_episodes,

        # Reward statistics
        'reward_mean': float(np.mean(episode_rewards)),
        'reward_std': float(np.std(episode_rewards)),
        'reward_se': float(np.std(episode_rewards) / np.sqrt(num_episodes)),
        'reward_median': float(np.median(episode_rewards)),
        'reward_min': float(np.min(episode_rewards)),
        'reward_max': float(np.max(episode_rewards)),

        # Handover statistics
        'handover_mean': float(np.mean(episode_handovers)),
        'handover_std': float(np.std(episode_handovers)),
        'handover_median': float(np.median(episode_handovers)),
        'handover_min': int(np.min(episode_handovers)),
        'handover_max': int(np.max(episode_handovers)),
        'handover_zero_episodes': int(sum(1 for h in episode_handovers if h == 0)),

        # Connectivity statistics
        'connectivity_mean': float(np.mean(episode_connectivity)),
        'connectivity_std': float(np.std(episode_connectivity)),
        'connectivity_median': float(np.median(episode_connectivity)),
        'connectivity_min': float(np.min(episode_connectivity)),
        'connectivity_max': float(np.max(episode_connectivity)),

        # RSRP statistics
        'rsrp_mean': float(np.mean(episode_avg_rsrp)),
        'rsrp_std': float(np.std(episode_avg_rsrp)),

        # Episode length
        'episode_length_mean': float(np.mean(episode_lengths)),
        'episode_length_std': float(np.std(episode_lengths)),

        # Raw data (for further analysis)
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_handovers': [int(h) for h in episode_handovers],
        'episode_connectivity': [float(c) for c in episode_connectivity],
    }

    return results


def print_results(results):
    """Print formatted results"""
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}\n")

    print(f"Seed: {results['seed']}")
    print(f"Episodes: {results['num_episodes']}\n")

    print(f"📊 Reward: {results['reward_mean']:,.1f} ± {results['reward_std']:.1f}")
    print(f"   Median: {results['reward_median']:,.1f}")
    print(f"   Range: [{results['reward_min']:,.1f}, {results['reward_max']:,.1f}]\n")

    print(f"🔄 Handovers: {results['handover_mean']:.2f} ± {results['handover_std']:.2f} per episode")
    print(f"   Median: {results['handover_median']:.1f}")
    print(f"   Range: [{results['handover_min']}, {results['handover_max']}]")
    print(f"   Zero handovers: {results['handover_zero_episodes']}/{results['num_episodes']} episodes\n")

    print(f"📡 Connectivity: {results['connectivity_mean']*100:.1f}% ± {results['connectivity_std']*100:.1f}%")
    print(f"   Median: {results['connectivity_median']*100:.1f}%")
    print(f"   Range: [{results['connectivity_min']*100:.1f}%, {results['connectivity_max']*100:.1f}%]\n")

    print(f"📶 RSRP (dBm): {results['rsrp_mean']:.1f} ± {results['rsrp_std']:.1f}\n")

    print(f"⏱️  Episode Length: {results['episode_length_mean']:.1f} ± {results['episode_length_std']:.1f} steps")


def aggregate_multi_seed_results(all_results):
    """Aggregate results across multiple seeds"""

    # Flatten all episode data
    all_rewards = []
    all_handovers = []
    all_connectivity = []

    for results in all_results:
        all_rewards.extend(results['episode_rewards'])
        all_handovers.extend(results['episode_handovers'])
        all_connectivity.extend(results['episode_connectivity'])

    # Per-seed means
    seed_reward_means = [r['reward_mean'] for r in all_results]
    seed_handover_means = [r['handover_mean'] for r in all_results]
    seed_connectivity_means = [r['connectivity_mean'] for r in all_results]

    aggregated = {
        'num_seeds': len(all_results),
        'total_episodes': sum(r['num_episodes'] for r in all_results),

        # Reward (pooled across all episodes)
        'reward_mean': float(np.mean(all_rewards)),
        'reward_std': float(np.std(all_rewards)),
        'reward_se': float(np.std(seed_reward_means) / np.sqrt(len(all_results))),  # SE across seeds
        'reward_seed_std': float(np.std(seed_reward_means)),  # Variance between seeds

        # Handovers
        'handover_mean': float(np.mean(all_handovers)),
        'handover_std': float(np.std(all_handovers)),
        'handover_se': float(np.std(seed_handover_means) / np.sqrt(len(all_results))),
        'handover_zero_pct': float(sum(1 for h in all_handovers if h == 0) / len(all_handovers) * 100),

        # Connectivity
        'connectivity_mean': float(np.mean(all_connectivity)),
        'connectivity_std': float(np.std(all_connectivity)),
        'connectivity_se': float(np.std(seed_connectivity_means) / np.sqrt(len(all_results))),

        # Per-seed statistics
        'per_seed_results': all_results,
    }

    return aggregated


def main():
    """Main evaluation pipeline"""
    parser = argparse.ArgumentParser(description='Comprehensive DQN Evaluation')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file')
    parser.add_argument('--num-episodes', type=int, default=100, help='Episodes per model')
    parser.add_argument('--output-dir', type=str, default='results/dqn_comprehensive', help='Output directory')
    args = parser.parse_args()

    print("="*80)
    print("DQN COMPREHENSIVE EVALUATION")
    print("="*80)
    print("\nEvaluating trained DQN models on full metrics suite")
    print("Metrics: Reward, Handovers, Connectivity, RSRP, QoS")
    print(f"\nConfig: {args.config}")
    print(f"Episodes per model: {args.num_episodes}")
    print("\n" + "="*80 + "\n")

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load satellites
    satellite_ids, metadata = load_stage4_optimized_satellites(
        constellation_filter='starlink',
        return_metadata=True,
        use_rl_training_data=False
    )
    print(f"Loaded {len(satellite_ids)} satellites\n")

    # Create environment
    adapter = AdapterWrapper(config)
    env = SatelliteHandoverEnv(adapter, satellite_ids, config)

    # Find all trained models
    model_paths = {
        42: "output/academic_experiment_20251217/seed_42/models/dqn_final.zip",
        123: "output/academic_experiment_20251217/seed_123/models/dqn_final.zip",
        456: "output/academic_experiment_20251217/seed_456/models/dqn_final.zip",
        789: "output/academic_experiment_20251217/seed_789/models/dqn_final.zip",
        2024: "output/academic_experiment_20251217/seed_2024/models/dqn_final.zip",
    }

    # Evaluate each model
    all_results = []
    for seed, model_path in model_paths.items():
        if Path(model_path).exists():
            results = evaluate_dqn_model(model_path, env, num_episodes=args.num_episodes, seed=seed)
            print_results(results)
            all_results.append(results)
        else:
            print(f"⚠️  Model not found: {model_path}")

    # Aggregate results
    if len(all_results) > 0:
        print("\n" + "="*80)
        print("AGGREGATED RESULTS (ALL SEEDS)")
        print("="*80 + "\n")

        aggregated = aggregate_multi_seed_results(all_results)

        print(f"Total seeds evaluated: {aggregated['num_seeds']}")
        print(f"Total episodes: {aggregated['total_episodes']}\n")

        print(f"📊 Reward: {aggregated['reward_mean']:,.1f} ± {aggregated['reward_se']:.1f} (SE across seeds)")
        print(f"   Pooled std: {aggregated['reward_std']:.1f}")
        print(f"   Seed-to-seed variation: {aggregated['reward_seed_std']:.1f}\n")

        print(f"🔄 Handovers: {aggregated['handover_mean']:.2f} ± {aggregated['handover_se']:.2f} per episode")
        print(f"   Pooled std: {aggregated['handover_std']:.2f}")
        print(f"   Zero handover episodes: {aggregated['handover_zero_pct']:.1f}%\n")

        print(f"📡 Connectivity: {aggregated['connectivity_mean']*100:.1f}% ± {aggregated['connectivity_se']*100:.1f}%")
        print(f"   Pooled std: {aggregated['connectivity_std']*100:.1f}%\n")

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / 'dqn_comprehensive_results.json'
        with open(output_file, 'w') as f:
            json.dump(aggregated, f, indent=2)

        print(f"✅ Results saved to: {output_file}")

        # Compare with baselines
        print("\n" + "="*80)
        print("COMPARISON WITH BASELINES")
        print("="*80 + "\n")

        # Load baseline results
        baseline_file = Path('results/baselines/baseline_results.json')
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                baselines = json.load(f)

            print(f"{'Policy':<20} {'Reward':>12} {'Handovers':>12} {'Connectivity':>12}")
            print("-" * 60)

            # DQN
            print(f"{'DQN (ours)':<20} {aggregated['reward_mean']:>12,.0f} "
                  f"{aggregated['handover_mean']:>12.2f} "
                  f"{aggregated['connectivity_mean']*100:>11.1f}%")

            # Baselines
            for name, stats in baselines.items():
                display_name = name.replace('_', ' ').title()
                print(f"{display_name:<20} {stats['reward_mean']:>12,.0f} "
                      f"{stats['handover_mean']:>12.2f} "
                      f"{stats['connectivity_mean']*100:>11.1f}%")

            # Key insights
            print("\n" + "="*80)
            print("KEY FINDINGS")
            print("="*80 + "\n")

            dqn_reward = aggregated['reward_mean']
            dqn_handovers = aggregated['handover_mean']
            dqn_connectivity = aggregated['connectivity_mean']

            print("1. REWARD COMPARISON:")
            for name, stats in baselines.items():
                diff_pct = (stats['reward_mean'] / dqn_reward - 1) * 100
                display_name = name.replace('_', ' ').title()
                print(f"   {display_name}: {diff_pct:+.1f}% vs DQN")

            print("\n2. HANDOVER EFFICIENCY:")
            print(f"   DQN: {dqn_handovers:.2f} handovers/episode")
            for name, stats in baselines.items():
                display_name = name.replace('_', ' ').title()
                print(f"   {display_name}: {stats['handover_mean']:.2f} handovers/episode")

            print("\n3. CONNECTIVITY:")
            print(f"   DQN: {dqn_connectivity*100:.1f}%")
            for name, stats in baselines.items():
                display_name = name.replace('_', ' ').title()
                print(f"   {display_name}: {stats['connectivity_mean']*100:.1f}%")

            print("\n4. ANALYSIS:")
            if dqn_handovers < baselines['random']['handover_mean']:
                print(f"   ✅ DQN performs {baselines['random']['handover_mean']/dqn_handovers:.1f}x fewer handovers than Random policy")
            if dqn_connectivity > baselines.get('max_rvt', {}).get('connectivity_mean', 0):
                print("   ✅ DQN maintains better connectivity than Max RVT")
            if dqn_reward < baselines['random']['reward_mean']:
                print("   ⚠️  DQN reward still lower than simple baselines")
                print("   💡 Suggests reward function needs rebalancing")

            # New insights based on handover efficiency
            if dqn_handovers < 3.0:
                print(f"   ✅ DQN shows good handover restraint ({dqn_handovers:.2f}/episode)")
            if dqn_connectivity > 0.85:
                print(f"   ✅ DQN maintains strong connectivity ({dqn_connectivity*100:.1f}%)")

        else:
            print("⚠️  Baseline results not found. Run evaluate_baselines.py first.")

    else:
        print("❌ No models were successfully evaluated")


if __name__ == '__main__':
    main()
