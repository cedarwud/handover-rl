#!/usr/bin/env python3
"""
Evaluate Baseline Policies

Evaluates simple heuristic policies against the DQN learned policy.
Provides performance comparison for academic research.

Usage:
    python evaluate_baselines.py --config configs/config.yaml --num-episodes 100
"""

import sys
sys.path.insert(0, 'src')

import argparse
import yaml
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import warnings
import logging

from baselines import get_baseline_policy, BASELINE_POLICIES
from utils.satellite_utils import load_stage4_optimized_satellites
from environments import SatelliteHandoverEnv
from adapters.adapter_wrapper import AdapterWrapper

warnings.filterwarnings('ignore')
logging.disable(logging.INFO)


def evaluate_policy(policy, env, num_episodes: int = 100):
    """Evaluate a policy on the environment
    
    Args:
        policy: Policy to evaluate (baseline or DQN)
        env: Environment
        num_episodes: Number of episodes to evaluate
        
    Returns:
        dict: Statistics including rewards, handovers, blocking, etc.
    """
    episode_rewards = []
    episode_handovers = []
    episode_lengths = []
    episode_avg_rsrp = []
    episode_connectivity = []
    
    for ep in tqdm(range(num_episodes), desc=f"Evaluating {getattr(policy, 'name', 'Policy')}"):
        policy.reset() if hasattr(policy, 'reset') else None
        
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0
        rsrp_sum = 0
        
        while not done:
            # Get action from policy
            if hasattr(policy, 'select_action'):
                # Baseline policy
                action = policy.select_action(obs)
            else:
                # DQN policy
                action, _ = policy.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_length += 1
            
            # Track RSRP (current satellite is first in observation)
            if len(obs) > 0:
                rsrp_sum += obs[0, 10]  # RSRP at index 10
        
        episode_rewards.append(ep_reward)
        episode_handovers.append(env.episode_stats['num_handovers'])
        episode_lengths.append(ep_length)
        episode_avg_rsrp.append(rsrp_sum / ep_length if ep_length > 0 else 0)
        episode_connectivity.append(env.episode_stats['connected_steps'] / ep_length if ep_length > 0 else 0)
    
    # Calculate statistics
    return {
        'reward_mean': float(np.mean(episode_rewards)),
        'reward_std': float(np.std(episode_rewards)),
        'reward_se': float(np.std(episode_rewards) / np.sqrt(len(episode_rewards))),
        'handover_mean': float(np.mean(episode_handovers)),
        'handover_std': float(np.std(episode_handovers)),
        'rsrp_mean': float(np.mean(episode_avg_rsrp)),
        'rsrp_std': float(np.std(episode_avg_rsrp)),
        'connectivity_mean': float(np.mean(episode_connectivity)),
        'connectivity_std': float(np.std(episode_connectivity)),
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_handovers': [int(h) for h in episode_handovers],
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Baseline Policies')
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--num-episodes', type=int, default=100, help='Episodes to evaluate')
    parser.add_argument('--output-dir', type=str, default='results/baselines', help='Output directory')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Baseline Policy Evaluation")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Output: {output_dir}")
    print()
    
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
    
    # Evaluate all baseline policies
    results = {}
    
    for policy_name in ['random', 'always_stay', 'max_rsrp', 'max_elevation', 'max_rvt']:
        print(f"\n{'=' * 80}")
        print(f"Evaluating: {policy_name.upper()}")
        print('=' * 80)
        
        policy = get_baseline_policy(policy_name)
        stats = evaluate_policy(policy, env, args.num_episodes)
        results[policy_name] = stats
        
        print(f"\nResults:")
        print(f"  Reward: {stats['reward_mean']:,.1f} ± {stats['reward_std']:,.1f}")
        print(f"  Handovers: {stats['handover_mean']:.2f} ± {stats['handover_std']:.2f}")
        print(f"  RSRP: {stats['rsrp_mean']:.1f} dBm ± {stats['rsrp_std']:.1f}")
        print(f"  Connectivity: {stats['connectivity_mean']*100:.1f}%")
    
    # Save results
    results_file = output_dir / 'baseline_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"✅ Results saved to: {results_file}")
    print('=' * 80)


if __name__ == '__main__':
    main()
