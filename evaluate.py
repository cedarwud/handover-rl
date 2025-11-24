#!/usr/bin/env python3
"""
Model Evaluation Script

Evaluate trained RL agents and compare with baseline strategies.

Usage:
    # Evaluate DQN model vs RSRP Baseline
    python evaluate.py --model output/dqn_level1/checkpoints/best_model.pth \
                       --algorithm dqn \
                       --episodes 20 \
                       --output-dir evaluation/dqn_vs_baseline

    # Evaluate multiple checkpoints
    python evaluate.py --model output/dqn_level1/checkpoints/checkpoint_ep50.pth \
                       --algorithm dqn \
                       --episodes 20

Metrics Collected:
- Total reward per episode
- Average reward over all episodes
- Handover count
- Ping-pong events
- Average RSRP
- Episode length
- Standard deviation of rewards

Comparison:
- DQN Agent (trained model)
- RSRP Baseline (greedy RSRP selection)
- Reward improvement percentage
"""

import sys
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import framework components
from adapters import AdapterWrapper  # NEW v3.0: Unified adapter
from environments.satellite_handover_env import SatelliteHandoverEnv
from agents import DQNAgent, RSRPBaselineAgent
from utils.satellite_utils import load_stage4_optimized_satellites


# ========== Setup Functions ==========

def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'evaluation.log'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


# ========== Evaluation Functions ==========

def evaluate_agent(env, agent, num_episodes, start_time, logger, agent_name="Agent"):
    """
    Evaluate agent over multiple episodes

    Args:
        env: Environment instance
        agent: Agent to evaluate
        num_episodes: Number of episodes to run
        start_time: Start time for episodes
        logger: Logger instance
        agent_name: Name for logging

    Returns:
        metrics: Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_handovers = []
    episode_ping_pongs = []
    episode_avg_rsrp = []
    episode_lengths = []

    episode_duration_minutes = 95  # Starlink orbital period

    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating {agent_name}")
    logger.info(f"{'='*80}")

    successful_episodes = 0
    attempted_episodes = 0
    max_attempts_per_episode = 20  # Max attempts to find valid start time
    skipped_episodes = 0

    pbar = tqdm(total=num_episodes, desc=f"Eval {agent_name}")

    while successful_episodes < num_episodes and attempted_episodes < num_episodes * max_attempts_per_episode:
        # Calculate episode start time
        time_offset_minutes = attempted_episodes * episode_duration_minutes
        episode_start_time = start_time + timedelta(minutes=time_offset_minutes)

        # Try to reset environment
        obs, info = env.reset(options={'start_time': episode_start_time})

        # Check if episode has valid initial state (at least one visible satellite)
        num_visible = info.get('num_visible', 0)

        if num_visible == 0:
            # No satellites visible at start - skip this episode
            attempted_episodes += 1
            skipped_episodes += 1
            logger.debug(f"Skipped episode at {episode_start_time.isoformat()} (no visible satellites)")
            continue

        attempted_episodes += 1

        # Episode loop
        episode_reward = 0
        step_count = 0
        rsrp_sum = 0
        done = False

        while not done:
            # Select action (deterministic for evaluation)
            action = agent.select_action(obs, deterministic=True)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Accumulate metrics
            episode_reward += reward
            step_count += 1

            # Track RSRP (from observation, feature 0 is RSRP)
            current_rsrp = obs[0, 0]  # Current satellite's RSRP
            rsrp_sum += current_rsrp

            obs = next_obs

        # Episode statistics
        avg_rsrp = rsrp_sum / step_count if step_count > 0 else 0
        # Use correct keys from environment's episode_stats
        handovers = info.get('num_handovers', 0)
        ping_pongs = info.get('num_ping_pongs', 0)

        episode_rewards.append(episode_reward)
        episode_handovers.append(handovers)
        episode_ping_pongs.append(ping_pongs)
        episode_avg_rsrp.append(avg_rsrp)
        episode_lengths.append(step_count)

        # Increment successful episode counter
        successful_episodes += 1
        pbar.update(1)

    pbar.close()

    # Log summary of episode selection
    logger.info(f"\nEpisode Selection Summary:")
    logger.info(f"  Successful episodes: {successful_episodes}")
    logger.info(f"  Attempted episodes: {attempted_episodes}")
    logger.info(f"  Skipped episodes (no satellites): {skipped_episodes}")

    # Compute aggregate metrics
    metrics = {
        'agent_name': agent_name,
        'num_episodes': num_episodes,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'mean_handovers': float(np.mean(episode_handovers)),
        'std_handovers': float(np.std(episode_handovers)),
        'mean_ping_pongs': float(np.mean(episode_ping_pongs)),
        'mean_avg_rsrp': float(np.mean(episode_avg_rsrp)),
        'mean_episode_length': float(np.mean(episode_lengths)),
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_handovers': [int(h) for h in episode_handovers],
        'episode_ping_pongs': [int(p) for p in episode_ping_pongs],
    }

    logger.info(f"\n{agent_name} Results:")
    logger.info(f"  Mean Reward: {metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f}")
    logger.info(f"  Reward Range: [{metrics['min_reward']:.3f}, {metrics['max_reward']:.3f}]")
    logger.info(f"  Mean Handovers: {metrics['mean_handovers']:.2f} ± {metrics['std_handovers']:.2f}")
    logger.info(f"  Mean Ping-Pongs: {metrics['mean_ping_pongs']:.2f}")
    logger.info(f"  Mean Avg RSRP: {metrics['mean_avg_rsrp']:.2f} dBm")

    return metrics


def compare_agents(dqn_metrics, baseline_metrics, logger):
    """
    Compare DQN agent with baseline

    Args:
        dqn_metrics: DQN evaluation metrics
        baseline_metrics: Baseline evaluation metrics
        logger: Logger instance

    Returns:
        comparison: Dictionary with comparison results
    """
    # Calculate absolute improvement (always valid)
    absolute_improvement = dqn_metrics['mean_reward'] - baseline_metrics['mean_reward']

    # Calculate improvement percentages (only if baseline is not near zero)
    if abs(baseline_metrics['mean_reward']) > 0.5:
        reward_improvement = (absolute_improvement / abs(baseline_metrics['mean_reward']) * 100)
        use_percentage = True
    else:
        # Baseline too close to zero - percentage is misleading
        reward_improvement = None
        use_percentage = False

    # Handle division by zero for handover reduction
    if baseline_metrics['mean_handovers'] > 0:
        handover_reduction = ((baseline_metrics['mean_handovers'] - dqn_metrics['mean_handovers']) /
                              baseline_metrics['mean_handovers'] * 100)
    else:
        handover_reduction = 0.0

    # Handle division by zero for ping-pong reduction
    if baseline_metrics['mean_ping_pongs'] > 0:
        ping_pong_reduction = ((baseline_metrics['mean_ping_pongs'] - dqn_metrics['mean_ping_pongs']) /
                               baseline_metrics['mean_ping_pongs'] * 100)
    else:
        ping_pong_reduction = 0.0

    comparison = {
        'reward_improvement_absolute': float(absolute_improvement),
        'reward_improvement_percent': float(reward_improvement) if reward_improvement is not None else None,
        'handover_reduction_percent': float(handover_reduction),
        'ping_pong_reduction_percent': float(ping_pong_reduction),
        'dqn_mean_reward': dqn_metrics['mean_reward'],
        'baseline_mean_reward': baseline_metrics['mean_reward'],
        'dqn_mean_handovers': dqn_metrics['mean_handovers'],
        'baseline_mean_handovers': baseline_metrics['mean_handovers'],
    }

    logger.info(f"\n{'='*80}")
    logger.info("DQN vs RSRP Baseline Comparison")
    logger.info(f"{'='*80}")

    # Display reward improvement
    if use_percentage:
        logger.info(f"Reward Improvement: {reward_improvement:+.2f}%")
    else:
        logger.info(f"Reward Improvement (absolute): {absolute_improvement:+.3f}")
        logger.info(f"  (Baseline too close to zero - percentage misleading)")

    logger.info(f"  DQN:      {dqn_metrics['mean_reward']:.3f} ± {dqn_metrics['std_reward']:.3f}")
    logger.info(f"  Baseline: {baseline_metrics['mean_reward']:.3f} ± {baseline_metrics['std_reward']:.3f}")
    logger.info(f"  Absolute Gap: {absolute_improvement:+.3f}")

    logger.info(f"\nHandover Reduction: {handover_reduction:+.2f}%")
    logger.info(f"  DQN:      {dqn_metrics['mean_handovers']:.2f}")
    logger.info(f"  Baseline: {baseline_metrics['mean_handovers']:.2f}")
    logger.info(f"\nPing-Pong Reduction: {ping_pong_reduction:+.2f}%")
    logger.info(f"  DQN:      {dqn_metrics['mean_ping_pongs']:.2f}")
    logger.info(f"  Baseline: {baseline_metrics['mean_ping_pongs']:.2f}")

    # Phase 1b goal check
    logger.info(f"\n{'='*80}")
    if use_percentage and reward_improvement > 20:
        logger.info("✅ Phase 1b Goal ACHIEVED: Reward improvement > +20%")
    elif use_percentage:
        logger.info(f"⚠️  Phase 1b Goal NOT YET: Reward improvement = {reward_improvement:+.2f}% (target: >+20%)")
    else:
        # Use absolute improvement for goal check
        if absolute_improvement > 0.5:
            logger.info(f"✅ Phase 1b Goal ACHIEVED: Absolute improvement = {absolute_improvement:+.3f} (significantly better than baseline)")
        else:
            logger.info(f"⚠️  Phase 1b Goal NOT YET: Absolute improvement = {absolute_improvement:+.3f} (target: >+0.5)")
    logger.info(f"{'='*80}")

    return comparison


def save_evaluation_report(output_dir, dqn_metrics, baseline_metrics, comparison, config):
    """
    Save evaluation report as JSON

    Args:
        output_dir: Output directory
        dqn_metrics: DQN evaluation metrics
        baseline_metrics: Baseline evaluation metrics
        comparison: Comparison results
        config: Configuration dictionary
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'dqn_metrics': dqn_metrics,
        'baseline_metrics': baseline_metrics,
        'comparison': comparison,
    }

    report_path = output_dir / 'evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Evaluation report saved to: {report_path}")


# ========== Main Evaluation Function ==========

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained RL agent vs baselines',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model', type=str,
        required=True,
        help='Path to trained model checkpoint (e.g., output/dqn_level1/checkpoints/best_model.pth)'
    )

    parser.add_argument(
        '--algorithm', type=str,
        choices=['dqn'],
        default='dqn',
        help='RL algorithm type'
    )

    parser.add_argument(
        '--episodes', type=int,
        default=20,
        help='Number of evaluation episodes'
    )

    parser.add_argument(
        '--config', type=str,
        default='configs/data_gen_config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--output-dir', type=str,
        default='evaluation/dqn_vs_baseline',
        help='Output directory for evaluation results'
    )

    parser.add_argument(
        '--seed', type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)

    logger.info(f"{'='*80}")
    logger.info("Model Evaluation - DQN vs RSRP Baseline")
    logger.info(f"{'='*80}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Output: {args.output_dir}")

    # Load configuration
    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Add default configs if missing
    if 'environment' not in config:
        config['environment'] = {
            'time_step_seconds': 5,
            'episode_duration_minutes': 95,
            'max_visible_satellites': 10,
            'reward': {
                'qos_weight': 1.0,
                'handover_penalty': -0.1,
                'ping_pong_penalty': -0.2,
            }
        }

    if 'agent' not in config:
        config['agent'] = {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'batch_size': 64,
            'buffer_capacity': 10000,
            'target_update_freq': 100,
            'hidden_dim': 128,
        }

    # Initialize adapter
    logger.info("\nInitializing Orbit Adapter...")
    adapter = AdapterWrapper(config)  # Auto-selects precompute or real-time

    # Load satellite pool
    logger.info("Loading satellite pool from orbit-engine Stage 4...")
    satellite_ids, metadata = load_stage4_optimized_satellites(
        constellation_filter='starlink',
        return_metadata=True,
        use_rl_training_data=False,
        use_candidate_pool=False
    )

    logger.info(f"Loaded {len(satellite_ids)} satellites from optimized pool")

    # Create environment
    logger.info("Creating environment...")
    env = SatelliteHandoverEnv(adapter, satellite_ids, config)

    # ========== Load trained DQN agent ==========
    logger.info(f"\nLoading trained DQN model from: {args.model}")
    dqn_agent = DQNAgent(env.observation_space, env.action_space, config)
    dqn_agent.load(args.model)
    logger.info("✅ DQN model loaded")

    # ========== Create RSRP Baseline agent ==========
    logger.info("\nCreating RSRP Baseline agent...")
    baseline_agent = RSRPBaselineAgent(env.observation_space, env.action_space, config)
    logger.info("✅ RSRP Baseline agent created")

    # ========== Evaluate agents ==========
    # Use time range within precompute table (2025-10-10 to 2025-11-08)
    start_time = datetime(2025, 10, 10, 0, 0, 0)

    # Evaluate DQN
    dqn_metrics = evaluate_agent(
        env, dqn_agent, args.episodes, start_time, logger, agent_name="DQN Agent"
    )

    # Evaluate Baseline
    baseline_metrics = evaluate_agent(
        env, baseline_agent, args.episodes, start_time, logger, agent_name="RSRP Baseline"
    )

    # ========== Compare results ==========
    comparison = compare_agents(dqn_metrics, baseline_metrics, logger)

    # ========== Save report ==========
    eval_config = {
        'model_path': args.model,
        'algorithm': args.algorithm,
        'num_episodes': args.episodes,
        'seed': args.seed,
        'num_satellites': len(satellite_ids),
    }

    save_evaluation_report(output_dir, dqn_metrics, baseline_metrics, comparison, eval_config)

    logger.info(f"\n✅ Evaluation complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
