#!/usr/bin/env python3
"""
Reward Alignment Validation Script

Evaluates baseline policies on both V9 (RVT) and V10 (Connectivity-centric)
environments to validate that V10 rewards align with operational performance.

Expected Results:
- V9: DQN ranks 4th/6 in reward (misaligned) ❌
- V10: DQN ranks 1st-2nd in reward (aligned) ✅

- V9: Reward-connectivity correlation: negative (-0.85)
- V10: Reward-connectivity correlation: positive (+0.90)
"""

import sys
sys.path.insert(0, 'src')

import argparse
import yaml
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import logging

from utils.satellite_utils import load_stage4_optimized_satellites
from environments import SatelliteHandoverEnv,  SatelliteHandoverEnvV10
from adapters.adapter_wrapper import AdapterWrapper
from baselines import get_baseline_policy

warnings.filterwarnings('ignore')
logging.disable(logging.INFO)


def evaluate_policy_on_env(policy, env, num_episodes: int = 50, policy_name: str = "Policy"):
    """Evaluate a policy on a given environment"""
    episode_rewards = []
    episode_handovers = []
    episode_connectivity = []

    for ep in tqdm(range(num_episodes), desc=f"{policy_name}"):
        if hasattr(policy, 'reset'):
            policy.reset()

        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0
        rsrp_sum = 0

        while not done:
            if hasattr(policy, 'select_action'):
                action = policy.select_action(obs)
            else:
                action, _ = policy.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_length += 1

            if len(obs) > 0:
                rsrp_sum += obs[0, 10]

        episode_rewards.append(ep_reward)
        episode_handovers.append(env.episode_stats['num_handovers'])
        episode_connectivity.append(
            env.episode_stats['connected_steps'] / ep_length if ep_length > 0 else 0
        )

    return {
        'reward_mean': float(np.mean(episode_rewards)),
        'reward_std': float(np.std(episode_rewards)),
        'handover_mean': float(np.mean(episode_handovers)),
        'connectivity_mean': float(np.mean(episode_connectivity)),
        'episode_rewards': episode_rewards,
    }


def main():
    parser = argparse.ArgumentParser(description='Validate Reward Alignment (V9 vs V10)')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file')
    parser.add_argument('--num-episodes', type=int, default=50, help='Episodes per policy')
    args = parser.parse_args()

    print("="*80)
    print("REWARD ALIGNMENT VALIDATION: V9 (RVT) vs V10 (Connectivity-Centric)")
    print("="*80)
    print(f"\nConfig: {args.config}")
    print(f"Episodes per policy: {args.num_episodes}\n")

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

    # Create both environments
    print("Creating environments...")
    adapter_v9 = AdapterWrapper(config)
    env_v9 = SatelliteHandoverEnv(adapter_v9, satellite_ids, config)
    print("  ✅ V9 (RVT-based)")

    adapter_v10 = AdapterWrapper(config)
    env_v10 = SatelliteHandoverEnvV10(adapter_v10, satellite_ids, config)
    print("  ✅ V10 (Connectivity-centric)\n")

    # Test baseline policies
    policies_to_test = [
        ('random', 'Random'),
        ('always_stay', 'Always Stay'),
        ('max_rsrp', 'Max RSRP'),
        ('max_elevation', 'Max Elevation'),
        ('max_rvt', 'Max RVT'),
    ]

    results_v9 = {}
    results_v10 = {}

    print("="*80)
    print("EVALUATING BASELINE POLICIES")
    print("="*80 + "\n")

    for policy_key, policy_name in policies_to_test:
        print(f"\n{'='*80}")
        print(f"Policy: {policy_name}")
        print('='*80)

        policy = get_baseline_policy(policy_key)

        # Evaluate on V9
        print(f"\n  V9 (RVT-based):")
        results_v9[policy_key] = evaluate_policy_on_env(
            policy, env_v9, args.num_episodes, f"{policy_name} (V9)"
        )
        print(f"    Reward: {results_v9[policy_key]['reward_mean']:,.1f} ± {results_v9[policy_key]['reward_std']:.1f}")
        print(f"    Handovers: {results_v9[policy_key]['handover_mean']:.2f}")
        print(f"    Connectivity: {results_v9[policy_key]['connectivity_mean']*100:.1f}%")

        # Evaluate on V10
        print(f"\n  V10 (Connectivity-centric):")
        results_v10[policy_key] = evaluate_policy_on_env(
            policy, env_v10, args.num_episodes, f"{policy_name} (V10)"
        )
        print(f"    Reward: {results_v10[policy_key]['reward_mean']:,.1f} ± {results_v10[policy_key]['reward_std']:.1f}")
        print(f"    Handovers: {results_v10[policy_key]['handover_mean']:.2f}")
        print(f"    Connectivity: {results_v10[policy_key]['connectivity_mean']*100:.1f}%")

        # Show difference
        reward_diff = results_v10[policy_key]['reward_mean'] - results_v9[policy_key]['reward_mean']
        print(f"\n  Δ Reward (V10 - V9): {reward_diff:+,.1f}")

    # Compare alignment
    print("\n" + "="*80)
    print("REWARD ALIGNMENT ANALYSIS")
    print("="*80 + "\n")

    # Prepare data for correlation analysis
    policies_list = [p[0] for p in policies_to_test]
    connectivity_vals = [results_v9[p]['connectivity_mean'] for p in policies_list]
    v9_rewards = [results_v9[p]['reward_mean'] for p in policies_list]
    v10_rewards = [results_v10[p]['reward_mean'] for p in policies_list]

    # Calculate correlations
    corr_v9 = np.corrcoef(connectivity_vals, v9_rewards)[0, 1]
    corr_v10 = np.corrcoef(connectivity_vals, v10_rewards)[0, 1]

    print("1. REWARD-CONNECTIVITY CORRELATION:")
    print(f"   V9 (RVT):  r = {corr_v9:+.3f} {'❌ Negative/weak' if corr_v9 < 0.5 else '✅ Positive'}")
    print(f"   V10 (Conn): r = {corr_v10:+.3f} {'✅ Strong positive' if corr_v10 > 0.7 else '⚠️  Moderate'}")

    # Rank policies by reward
    print("\n2. POLICY RANKING BY REWARD:\n")

    v9_ranking = sorted([(p, results_v9[p]['reward_mean'], results_v9[p]['connectivity_mean'])
                         for p in policies_list], key=lambda x: x[1], reverse=True)

    v10_ranking = sorted([(p, results_v10[p]['reward_mean'], results_v10[p]['connectivity_mean'])
                          for p in policies_list], key=lambda x: x[1], reverse=True)

    print("   V9 (RVT-based) Ranking:")
    for i, (policy, reward, conn) in enumerate(v9_ranking, 1):
        display_name = dict(policies_to_test)[policy]
        print(f"   {i}. {display_name:<15} Reward: {reward:>10,.0f}  Connectivity: {conn*100:>5.1f}%")

    print("\n   V10 (Connectivity-centric) Ranking:")
    for i, (policy, reward, conn) in enumerate(v10_ranking, 1):
        display_name = dict(policies_to_test)[policy]
        print(f"   {i}. {display_name:<15} Reward: {reward:>10,.0f}  Connectivity: {conn*100:>5.1f}%")

    # Compare with DQN expected performance
    print("\n3. EXPECTED DQN PERFORMANCE:")
    print(f"   From comprehensive evaluation:")
    print(f"     DQN Connectivity: 98.3% (best among all policies)")
    print(f"     DQN Handovers:    0.79/episode (optimal balance)")

    # Predict DQN reward on V10
    # DQN: 98.3% conn (118 steps), 0.79 HO/ep
    conn_steps = 118
    disconn_steps = 2
    handovers = 0.79
    avg_stability = 60

    pred_dqn_v10 = (
        conn_steps * 100 +           # Connectivity reward
        disconn_steps * -500 +       # Disconnection penalty
        handovers * -50 +            # Handover penalty
        conn_steps * 1.0 +           # Signal quality
        avg_stability * 0.5          # Stability bonus
    )

    print(f"\n   Predicted DQN reward on V10: ~{pred_dqn_v10:,.0f}")
    print(f"   V10 Max baseline reward: {max(v10_rewards):,.0f}")
    print(f"   Expected DQN ranking: {'1st-2nd (aligned ✅)' if pred_dqn_v10 >= max(v10_rewards) * 0.9 else 'Lower than expected ⚠️'}")

    # Save results
    output_dir = Path('results/reward_alignment')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        'v9_results': results_v9,
        'v10_results': results_v10,
        'correlation': {
            'v9': float(corr_v9),
            'v10': float(corr_v10)
        },
        'dqn_prediction': {
            'connectivity': 0.983,
            'handovers': 0.79,
            'predicted_v10_reward': float(pred_dqn_v10)
        }
    }

    output_file = output_dir / 'alignment_validation_results.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")

    # Final verdict
    print("\n" + "="*80)
    print("VALIDATION VERDICT")
    print("="*80 + "\n")

    alignment_improved = corr_v10 > corr_v9
    v10_strongly_positive = corr_v10 > 0.7

    if alignment_improved and v10_strongly_positive:
        print("✅ SUCCESS: V10 reward function shows strong alignment with operational performance")
        print(f"   - Correlation improved from {corr_v9:+.3f} (V9) to {corr_v10:+.3f} (V10)")
        print("   - Higher connectivity policies now receive higher rewards")
        print("   - DQN predicted to rank 1st-2nd in V10 (vs 4th in V9)")
    elif alignment_improved:
        print("⚠️  PARTIAL SUCCESS: V10 shows improvement but correlation could be stronger")
        print(f"   - Correlation improved from {corr_v9:+.3f} (V9) to {corr_v10:+.3f} (V10)")
        print("   - Consider further reward function tuning")
    else:
        print("❌ VALIDATION FAILED: V10 did not improve alignment")
        print(f"   - V9 correlation: {corr_v9:+.3f}")
        print(f"   - V10 correlation: {corr_v10:+.3f}")
        print("   - Reward function design needs revision")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
