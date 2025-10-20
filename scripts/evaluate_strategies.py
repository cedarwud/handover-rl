#!/usr/bin/env python3
"""
Unified Strategy Evaluation Framework

Evaluates both RL agents and rule-based strategies using duck typing.
Only requirement: object must have select_action(observation) method.

Features:
- Supports RL agents (DQNAgent, etc.) and rule-based strategies
- Multi-level training integration
- Comprehensive metrics collection
- Comparison report generation
- TensorBoard logging (optional)

Usage:
    # Evaluate single strategy
    python scripts/evaluate_strategies.py \
        --strategy-type rule_based \
        --strategy-name a4_based \
        --level 1 \
        --episodes 100

    # Compare all strategies
    python scripts/evaluate_strategies.py \
        --compare-all \
        --level 1 \
        --output results/level1_comparison.csv

Academic Compliance:
- Real TLE data (via orbit-engine)
- Complete physics (ITU-R, 3GPP)
- Reproducible (seed-controlled)
"""

import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from adapters.orbit_engine_adapter import OrbitEngineAdapter
from environments.satellite_handover_env import SatelliteHandoverEnv
from agents import DQNAgent
from strategies import StrongestRSRPStrategy, A4BasedStrategy, D2BasedStrategy
from utils.satellite_utils import load_stage4_optimized_satellites
from configs import get_level_config

# TensorBoard (optional)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


# ========== Evaluation Functions ==========

def evaluate_strategy(
    strategy,
    env: SatelliteHandoverEnv,
    num_episodes: int = 100,
    seed: int = 42,
    start_time_base: datetime = None,
    episode_duration_minutes: int = 95,
    overlap_ratio: float = 0.5,
    logger: logging.Logger = None,
    tensorboard_writer = None
) -> Dict[str, Any]:
    """
    Evaluate strategy on environment.

    Args:
        strategy: Any object with select_action(observation) method
        env: SatelliteHandoverEnv instance
        num_episodes: Number of evaluation episodes
        seed: Random seed for reproducibility
        start_time_base: Base time for episode sampling
        episode_duration_minutes: Episode duration (default: 95 min)
        overlap_ratio: Episode overlap (default: 0.5)
        logger: Logger instance
        tensorboard_writer: TensorBoard writer (optional)

    Returns:
        metrics: Dict with performance metrics
            - avg_reward: Average episode reward
            - std_reward: Reward standard deviation
            - avg_handovers: Average handovers per episode
            - handover_rate_pct: Handover rate percentage
            - ping_pong_rate_pct: Ping-pong rate percentage
            - avg_rsrp_dbm: Average RSRP in dBm
            - episode_rewards: List of all episode rewards
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if start_time_base is None:
        start_time_base = datetime(2025, 7, 27, 0, 0, 0)

    # Calculate episode stride
    if overlap_ratio > 0:
        episode_stride_minutes = episode_duration_minutes * (1 - overlap_ratio)
    else:
        episode_stride_minutes = episode_duration_minutes

    # Metrics collection
    episode_rewards = []
    episode_handovers = []
    episode_ping_pongs = []
    episode_avg_rsrps = []
    episode_lengths = []

    logger.info(f"Evaluating strategy: {strategy.__class__.__name__}")
    logger.info(f"  Episodes: {num_episodes}")
    logger.info(f"  Seed: {seed}")

    for episode in tqdm(range(num_episodes), desc=f"Evaluating {strategy.__class__.__name__}"):
        # Time sampling (same as training)
        time_offset_minutes = episode * episode_stride_minutes
        episode_start_time = start_time_base + timedelta(minutes=time_offset_minutes)

        # Reset environment
        obs, info = env.reset(seed=seed + episode, options={'start_time': episode_start_time})

        episode_reward = 0.0
        episode_steps = 0
        done = False

        # Get current serving satellite index
        serving_idx = None
        if hasattr(env, 'current_serving_satellite_idx'):
            serving_idx = env.current_serving_satellite_idx

        # Episode loop
        while not done:
            # Select action (duck typing - works for both RL and rule-based)
            try:
                # For RL agents, use deterministic=True for evaluation
                if hasattr(strategy, 'select_action'):
                    action = strategy.select_action(
                        obs,
                        deterministic=True,
                        serving_satellite_idx=serving_idx
                    )
                else:
                    raise AttributeError(f"Strategy {strategy.__class__.__name__} has no select_action method")
            except TypeError:
                # Fallback for strategies that don't accept **kwargs
                action = strategy.select_action(obs)

            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update metrics
            episode_reward += reward
            episode_steps += 1
            obs = next_obs

            # Update serving satellite index
            if hasattr(env, 'current_serving_satellite_idx'):
                serving_idx = env.current_serving_satellite_idx

        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        episode_handovers.append(info.get('num_handovers', 0))
        episode_ping_pongs.append(info.get('num_ping_pongs', 0))
        episode_avg_rsrps.append(info.get('avg_rsrp', 0))

        # TensorBoard logging
        if tensorboard_writer:
            tensorboard_writer.add_scalar('Eval/Reward', episode_reward, episode)
            tensorboard_writer.add_scalar('Eval/Handovers', info.get('num_handovers', 0), episode)
            tensorboard_writer.add_scalar('Eval/PingPongs', info.get('num_ping_pongs', 0), episode)
            tensorboard_writer.add_scalar('Eval/AvgRSRP', info.get('avg_rsrp', 0), episode)

    # Calculate summary metrics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_handovers = np.mean(episode_handovers)
    avg_ping_pongs = np.mean(episode_ping_pongs)
    avg_rsrp = np.mean(episode_avg_rsrps)

    # Calculate rates
    avg_episode_length = np.mean(episode_lengths)
    handover_rate_pct = (avg_handovers / avg_episode_length) * 100 if avg_episode_length > 0 else 0

    # Ping-pong rate (as percentage of handovers)
    ping_pong_rate_pct = (avg_ping_pongs / avg_handovers * 100) if avg_handovers > 0 else 0

    metrics = {
        'strategy_name': strategy.__class__.__name__,
        'avg_reward': float(avg_reward),
        'std_reward': float(std_reward),
        'avg_handovers': float(avg_handovers),
        'handover_rate_pct': float(handover_rate_pct),
        'avg_ping_pongs': float(avg_ping_pongs),
        'ping_pong_rate_pct': float(ping_pong_rate_pct),
        'avg_rsrp_dbm': float(avg_rsrp),
        'avg_episode_length': float(avg_episode_length),
        'num_episodes': num_episodes,
        'episode_rewards': episode_rewards,
    }

    logger.info(f"\nEvaluation Results for {strategy.__class__.__name__}:")
    logger.info(f"  Avg Reward: {avg_reward:+.2f} ± {std_reward:.2f}")
    logger.info(f"  Avg Handovers: {avg_handovers:.2f}")
    logger.info(f"  Handover Rate: {handover_rate_pct:.2f}%")
    logger.info(f"  Ping-pong Rate: {ping_pong_rate_pct:.2f}%")
    logger.info(f"  Avg RSRP: {avg_rsrp:.1f} dBm")

    return metrics


def compare_strategies(
    strategies: Dict[str, Any],
    env: SatelliteHandoverEnv,
    num_episodes: int = 100,
    seed: int = 42,
    logger: logging.Logger = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Compare multiple strategies and return results table.

    Args:
        strategies: Dict of {name: strategy_object}
        env: Environment instance
        num_episodes: Evaluation episodes
        seed: Random seed
        logger: Logger instance
        output_path: Optional path to save CSV results

    Returns:
        df: DataFrame with comparison results
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    results = []

    logger.info(f"\n{'='*60}")
    logger.info(f"Comparing {len(strategies)} strategies")
    logger.info(f"{'='*60}\n")

    for name, strategy in strategies.items():
        logger.info(f"\n--- Evaluating: {name} ---")
        metrics = evaluate_strategy(
            strategy,
            env,
            num_episodes=num_episodes,
            seed=seed,
            logger=logger
        )
        metrics['strategy'] = name
        results.append(metrics)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by average reward (descending)
    df = df.sort_values('avg_reward', ascending=False)

    # Reorder columns for better readability
    column_order = [
        'strategy',
        'avg_reward',
        'std_reward',
        'avg_handovers',
        'handover_rate_pct',
        'avg_ping_pongs',
        'ping_pong_rate_pct',
        'avg_rsrp_dbm',
    ]

    # Only include columns that exist
    available_columns = [col for col in column_order if col in df.columns]
    df = df[available_columns]

    # Save to CSV if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"\n✅ Results saved to: {output_path}")

    return df


# ========== Main Function ==========

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate and compare handover strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Strategy selection
    parser.add_argument(
        '--strategy-type',
        type=str,
        choices=['rule_based', 'rl', 'all'],
        default='all',
        help='Type of strategy to evaluate'
    )

    parser.add_argument(
        '--strategy-name',
        type=str,
        choices=['strongest_rsrp', 'a4_based', 'd2_based', 'dqn'],
        help='Specific strategy to evaluate (if not --compare-all)'
    )

    # Comparison mode
    parser.add_argument(
        '--compare-all',
        action='store_true',
        help='Compare all available strategies'
    )

    # Training level
    parser.add_argument(
        '--level',
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        default=1,
        help='Training level (determines num_satellites and num_episodes)'
    )

    # Override episodes
    parser.add_argument(
        '--episodes',
        type=int,
        help='Override number of episodes (default: from level config)'
    )

    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='config/data_gen_config.yaml',
        help='Path to configuration file'
    )

    # Output
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for comparison results (CSV)'
    )

    # Reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Load configuration
    logger.info(f"Loading config from: {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Get training level config
    level_config = get_level_config(args.level)
    num_satellites = level_config['num_satellites']
    num_episodes = args.episodes if args.episodes else level_config['num_episodes']

    logger.info(f"\nEvaluation Configuration:")
    logger.info(f"  Level: {args.level} ({level_config['name']})")
    logger.info(f"  Satellites: {num_satellites}")
    logger.info(f"  Episodes: {num_episodes}")
    logger.info(f"  Seed: {args.seed}")

    # Initialize adapter
    logger.info("\nInitializing OrbitEngineAdapter...")
    adapter = OrbitEngineAdapter(config)

    # Load satellites
    logger.info("Loading satellite pool...")
    satellite_ids, metadata = load_stage4_optimized_satellites(
        constellation_filter='starlink',
        return_metadata=True
    )

    # Use subset based on level
    if num_satellites < len(satellite_ids):
        satellite_ids = satellite_ids[:num_satellites]

    logger.info(f"  Using {len(satellite_ids)} satellites")

    # Create environment
    logger.info("Creating environment...")
    env = SatelliteHandoverEnv(adapter, satellite_ids, config)

    # Create strategies
    strategies = {}

    if args.compare_all or args.strategy_type == 'all':
        # All strategies
        strategies['Strongest RSRP'] = StrongestRSRPStrategy()
        strategies['A4-based'] = A4BasedStrategy(threshold_dbm=-100.0, hysteresis_db=1.5)
        strategies['D2-based'] = D2BasedStrategy(threshold1_km=1412.8, threshold2_km=1005.8)
        logger.info("\nComparing all 3 rule-based strategies")

    elif args.strategy_name:
        # Single strategy
        if args.strategy_name == 'strongest_rsrp':
            strategies['Strongest RSRP'] = StrongestRSRPStrategy()
        elif args.strategy_name == 'a4_based':
            strategies['A4-based'] = A4BasedStrategy(threshold_dbm=-100.0, hysteresis_db=1.5)
        elif args.strategy_name == 'd2_based':
            strategies['D2-based'] = D2BasedStrategy(threshold1_km=1412.8, threshold2_km=1005.8)
        logger.info(f"\nEvaluating single strategy: {args.strategy_name}")

    else:
        logger.error("Must specify --strategy-name or --compare-all")
        return 1

    # Compare strategies
    output_path = Path(args.output) if args.output else None
    df = compare_strategies(
        strategies,
        env,
        num_episodes=num_episodes,
        seed=args.seed,
        logger=logger,
        output_path=output_path
    )

    # Print comparison table
    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON RESULTS")
    logger.info(f"{'='*80}\n")
    print(df.to_string(index=False))

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    best_strategy = df.iloc[0]
    logger.info(f"\n🏆 Best Strategy: {best_strategy['strategy']}")
    logger.info(f"   Avg Reward: {best_strategy['avg_reward']:+.2f}")
    logger.info(f"   Handover Rate: {best_strategy['handover_rate_pct']:.2f}%")
    logger.info(f"   Ping-pong Rate: {best_strategy['ping_pong_rate_pct']:.2f}%")

    return 0


if __name__ == '__main__':
    sys.exit(main())
