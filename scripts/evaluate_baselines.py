#!/usr/bin/env python3
"""
Evaluate Traditional Baseline Policies for Academic Comparison

This script evaluates traditional handover policies to compare with DQN:
1. RSRP Greedy: Always switch to satellite with highest RSRP (3GPP A3 event)
2. Random: Randomly select action at each step
3. Stay: Never handover (always action 0)

Academic Reference:
- RSRP-based is the standard baseline in LEO handover papers
- Provides lower bound for ML-based methods
"""

import sys
import logging
import numpy as np
from pathlib import Path
from scipy import stats
import yaml

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from environments import SatelliteHandoverEnv
from adapters import AdapterWrapper
from utils.satellite_utils import load_stage4_optimized_satellites

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class RSRPGreedyPolicy:
    """Always select satellite with highest RSRP"""
    def __init__(self, rsrp_index=0):
        self.rsrp_index = rsrp_index
        self.name = "RSRP Greedy"

    def predict(self, obs, deterministic=True):
        # obs shape: [K, 12] where K is max satellites
        rsrp_values = obs[:, self.rsrp_index]
        best_idx = np.argmax(rsrp_values)
        # If best is current (idx 0), stay; otherwise switch
        action = 0 if best_idx == 0 else best_idx
        return action, None


class RandomPolicy:
    """Randomly select action"""
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.name = "Random"

    def predict(self, obs, deterministic=True):
        action = np.random.randint(0, self.n_actions)
        return action, None


class StayPolicy:
    """Never handover - always stay with current satellite"""
    def __init__(self):
        self.name = "Stay (Never Handover)"

    def predict(self, obs, deterministic=True):
        return 0, None


def evaluate_policy(policy, env, num_episodes=100):
    """Evaluate a policy on the environment"""
    episode_handovers = []
    episode_rewards = []
    episode_qos = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        handovers = info.get('num_handovers', 0)
        avg_rsrp = info.get('avg_rsrp', 0)

        episode_handovers.append(handovers)
        episode_rewards.append(episode_reward)
        episode_qos.append(avg_rsrp)

        if (ep + 1) % 20 == 0:
            logger.info(f"  {policy.name}: {ep+1}/{num_episodes} episodes")

    return {
        'handovers': np.array(episode_handovers),
        'rewards': np.array(episode_rewards),
        'qos': np.array(episode_qos),
        'mean_handovers': np.mean(episode_handovers),
        'std_handovers': np.std(episode_handovers),
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
    }


def statistical_comparison(dqn_results, baseline_results, baseline_name):
    """Perform statistical comparison between DQN and baseline"""
    dqn_ho = dqn_results['handovers']
    base_ho = baseline_results['handovers']

    # Welch's t-test (unequal variances)
    t_stat, p_value = stats.ttest_ind(dqn_ho, base_ho, equal_var=False)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(dqn_ho)**2 + np.std(base_ho)**2) / 2)
    cohens_d = (np.mean(dqn_ho) - np.mean(base_ho)) / pooled_std if pooled_std > 0 else 0

    # Improvement percentage
    if np.mean(base_ho) > 0:
        improvement = (np.mean(base_ho) - np.mean(dqn_ho)) / np.mean(base_ho) * 100
    else:
        improvement = 0

    return {
        'baseline_name': baseline_name,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'improvement_percent': improvement,
        'significant': p_value < 0.05,
    }


def main():
    config_path = "configs/config.yaml"
    num_episodes = 100
    num_trials = 5  # Run multiple trials for statistical power

    print("=" * 80)
    print("BASELINE COMPARISON EVALUATION")
    print("=" * 80)
    print(f"\nConfig: {config_path}")
    print(f"Episodes per trial: {num_episodes}")
    print(f"Number of trials: {num_trials}")
    print()

    # Load config
    config = load_config(config_path)

    # Initialize environment
    logger.info("Creating evaluation environment...")
    adapter = AdapterWrapper(config)
    satellite_ids, _ = load_stage4_optimized_satellites(
        constellation_filter='starlink',
        return_metadata=True,
        use_rl_training_data=False,
        use_candidate_pool=False
    )

    env = SatelliteHandoverEnv(adapter, satellite_ids, config)

    # Initialize policies
    policies = {
        'rsrp_greedy': RSRPGreedyPolicy(rsrp_index=0),
        'random': RandomPolicy(n_actions=env.action_space.n),
        'stay': StayPolicy(),
    }

    # Results storage
    all_results = {name: [] for name in policies.keys()}

    # Run evaluations
    for trial in range(num_trials):
        print(f"\n{'='*80}")
        print(f"TRIAL {trial + 1}/{num_trials}")
        print("=" * 80)

        # Set different random seed for each trial
        np.random.seed(42 + trial * 100)

        for name, policy in policies.items():
            logger.info(f"Evaluating {policy.name}...")
            results = evaluate_policy(policy, env, num_episodes)
            all_results[name].append(results)
            print(f"  {policy.name}: {results['mean_handovers']:.2f} +/- {results['std_handovers']:.2f} handovers")

    # Aggregate results across trials
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS (Across All Trials)")
    print("=" * 80)

    aggregated = {}
    for name, trial_results in all_results.items():
        # Combine all episodes from all trials
        all_handovers = np.concatenate([r['handovers'] for r in trial_results])
        all_rewards = np.concatenate([r['rewards'] for r in trial_results])

        # Calculate mean of means (across trials)
        trial_means = [r['mean_handovers'] for r in trial_results]

        aggregated[name] = {
            'all_handovers': all_handovers,
            'all_rewards': all_rewards,
            'mean_handovers': np.mean(all_handovers),
            'std_handovers': np.std(all_handovers),
            'sem_handovers': np.std(trial_means) / np.sqrt(len(trial_means)),
            'trial_means': trial_means,
            'mean_of_means': np.mean(trial_means),
            'std_of_means': np.std(trial_means, ddof=1),
        }

    # Print aggregated results
    print(f"\n{'Method':<25} {'Mean HO':<12} {'SD':<10} {'SEM':<10} {'95% CI'}")
    print("-" * 80)

    for name, agg in aggregated.items():
        ci_low = agg['mean_of_means'] - 1.96 * agg['sem_handovers']
        ci_high = agg['mean_of_means'] + 1.96 * agg['sem_handovers']
        print(f"{policies[name].name:<25} {agg['mean_of_means']:.2f} +/- {agg['std_of_means']:.2f}  "
              f"{agg['std_handovers']:.2f}      {agg['sem_handovers']:.3f}     [{ci_low:.2f}, {ci_high:.2f}]")

    # Load DQN results for comparison
    print("\n" + "=" * 80)
    print("DQN vs BASELINE COMPARISON")
    print("=" * 80)

    # Read DQN results from our evaluation
    dqn_handovers = []
    for seed in [42, 123, 456, 789, 2024]:
        log_file = f"/tmp/eval_seed{seed}.log"
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            # Extract handover distribution
            import re
            for match in re.finditer(r'(\d+) handovers:\s+(\d+) episodes', content):
                ho_count = int(match.group(1))
                ep_count = int(match.group(2))
                dqn_handovers.extend([ho_count] * ep_count)
        except:
            pass

    if dqn_handovers:
        dqn_results = {
            'handovers': np.array(dqn_handovers),
            'mean_handovers': np.mean(dqn_handovers),
            'std_handovers': np.std(dqn_handovers),
        }

        print(f"\nDQN (SB3):           {dqn_results['mean_handovers']:.2f} +/- {dqn_results['std_handovers']:.2f} handovers")
        print()

        # Statistical comparison with each baseline
        print(f"\n{'Comparison':<30} {'Improvement':<15} {'p-value':<12} {'Cohens d':<12} {'Significant'}")
        print("-" * 80)

        for name, agg in aggregated.items():
            baseline_results = {'handovers': agg['all_handovers']}
            comparison = statistical_comparison(dqn_results, baseline_results, policies[name].name)

            sig_marker = "***" if comparison['p_value'] < 0.001 else ("**" if comparison['p_value'] < 0.01 else ("*" if comparison['p_value'] < 0.05 else ""))
            print(f"DQN vs {policies[name].name:<22} {comparison['improvement_percent']:>+.1f}%          "
                  f"{comparison['p_value']:.2e}     {comparison['cohens_d']:>+.2f}         {sig_marker}")

    # Save results
    output_file = "output/baseline_comparison.txt"
    Path("output").mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("Baseline Comparison Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Episodes per trial: {num_episodes}\n")
        f.write(f"Number of trials: {num_trials}\n")
        f.write(f"Total episodes evaluated: {num_episodes * num_trials}\n\n")

        f.write("Results (Mean +/- SD handovers):\n")
        for name, agg in aggregated.items():
            f.write(f"  {policies[name].name}: {agg['mean_of_means']:.2f} +/- {agg['std_of_means']:.2f}\n")

        if dqn_handovers:
            f.write(f"  DQN (SB3): {dqn_results['mean_handovers']:.2f} +/- {dqn_results['std_handovers']:.2f}\n")

    print(f"\nResults saved to: {output_file}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
