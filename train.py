#!/usr/bin/env python3
"""
Unified Training Entry Point - Modular RL Framework

Train RL agents for satellite handover optimization using the refactored
modular architecture.

Features:
- Algorithm-agnostic training (supports multiple RL algorithms)
- Multi-Level Training Strategy (Novel Aspect #1)
- Unified trainer interface (OffPolicyTrainer, OnPolicyTrainer)
- Complete integration with orbit-engine (real TLE data + physics)
- Reproducible experiments (seed-controlled)

Usage:
    # Quick validation (Level 1: 2 hours)
    python train.py --algorithm dqn --level 1 --output-dir output/quick_test

    # Full training (Level 5: 35 hours)
    python train.py --algorithm dqn --level 5 --output-dir output/full_training

    # With custom config
    python train.py --algorithm dqn --level 3 --config config/custom_config.yaml

Academic Compliance:
- Real TLE data from Space-Track.org
- Complete physics models (ITU-R, 3GPP)
- No simplified algorithms
- No mock data
- Reproducible (seed-controlled)
"""

import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import logging
import random
from tqdm import tqdm

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️  TensorBoard not available - metrics will not be logged to TensorBoard")

sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import framework components
from adapters.orbit_engine_adapter import OrbitEngineAdapter
from environments.satellite_handover_env import SatelliteHandoverEnv
from agents import DQNAgent
from trainers import OffPolicyTrainer
from utils.satellite_utils import load_stage4_optimized_satellites, verify_satellite_pool_integrity
from configs import get_level_config


# ========== Algorithm Registry ==========

ALGORITHM_REGISTRY = {
    'dqn': {
        'agent_class': DQNAgent,
        'trainer_class': OffPolicyTrainer,
        'description': 'Deep Q-Network (standard DQN)',
        'type': 'off-policy',
    },
    # Future algorithms can be added here
    # 'double_dqn': {...},
    # 'ppo': {...},
    # 'a2c': {...},
}


# ========== Setup Functions ==========

def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ========== Main Training Function ==========

def train(config, level_config, args, logger):
    """
    Main training loop

    Args:
        config: Base configuration dictionary (from YAML)
        level_config: Training level configuration
        args: Command line arguments
        logger: Logger instance
    """
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    # TensorBoard writer
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=str(log_dir))
        logger.info(f"TensorBoard logging to: {log_dir}")
    else:
        writer = None

    # Initialize adapter
    logger.info("Initializing OrbitEngineAdapter...")
    adapter = OrbitEngineAdapter(config)
    logger.info("✅ Adapter initialized")

    # Load satellite pool
    logger.info("=" * 80)
    logger.info("Loading optimized satellite pool from orbit-engine Stage 4...")
    logger.info("=" * 80)

    satellite_ids, metadata = load_stage4_optimized_satellites(
        constellation_filter='starlink',
        return_metadata=True
    )

    # Verify integrity
    is_valid, message = verify_satellite_pool_integrity(
        satellite_ids,
        expected_constellation='starlink',
        metadata=metadata
    )
    logger.info(message)

    # Use subset based on training level
    num_satellites = level_config['num_satellites']
    if num_satellites < len(satellite_ids):
        logger.info(f"\n⚙️  Training Level {args.level}: Using {num_satellites} satellites")
        satellite_ids = satellite_ids[:num_satellites]

    logger.info(f"\n📊 Final Satellite Pool:")
    logger.info(f"   Constellation: Starlink")
    logger.info(f"   Total: {len(satellite_ids)} satellites")
    logger.info(f"   Training Level: {args.level} ({level_config['name']})")
    logger.info(f"   Estimated Time: {level_config['estimated_time_hours']:.1f}h")
    logger.info(f"   First 5: {satellite_ids[:5]}")
    logger.info(f"   Last 5: {satellite_ids[-5:]}")
    logger.info(f"=" * 80 + "\n")

    # Create environment
    logger.info("Creating environment...")
    env = SatelliteHandoverEnv(adapter, satellite_ids, config)
    logger.info("✅ Environment created")

    # Get algorithm info
    algo_info = ALGORITHM_REGISTRY[args.algorithm]

    # Create agent
    logger.info(f"Creating {args.algorithm.upper()} agent...")
    AgentClass = algo_info['agent_class']
    agent = AgentClass(env.observation_space, env.action_space, config)
    logger.info("✅ Agent created")

    # Load checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        agent.load(args.resume)

    # Create trainer
    logger.info(f"Creating {algo_info['type']} trainer...")
    TrainerClass = algo_info['trainer_class']
    trainer = TrainerClass(env, agent, config)
    logger.info("✅ Trainer created")

    # Training metrics
    episode_rewards = []
    episode_handovers = []
    episode_avg_rsrp = []
    best_reward = -np.inf

    # Time configuration
    start_time_base = datetime(2025, 7, 27, 0, 0, 0)
    episode_duration_minutes = 95  # Starlink orbital period
    overlap_ratio = level_config['overlap']

    # Calculate episode stride
    if overlap_ratio > 0:
        episode_stride_minutes = episode_duration_minutes * (1 - overlap_ratio)
    else:
        episode_stride_minutes = episode_duration_minutes

    # Training loop
    num_episodes = level_config['num_episodes']
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting training: {num_episodes} episodes")
    logger.info(f"Algorithm: {args.algorithm.upper()}")
    logger.info(f"Training Level: {args.level} ({level_config['name']})")
    logger.info(f"{'='*80}\n")

    for episode in tqdm(range(num_episodes), desc="Training"):
        # Continuous time sampling with sliding window
        time_offset_minutes = episode * episode_stride_minutes
        episode_start_time = start_time_base + timedelta(minutes=time_offset_minutes)

        # Train one episode using trainer
        metrics = trainer.train_episode(
            episode_idx=episode,
            episode_start_time=episode_start_time,
            seed=args.seed + episode
        )

        # Record metrics
        episode_rewards.append(metrics['reward'])
        episode_handovers.append(metrics['handovers'])
        episode_avg_rsrp.append(metrics['avg_rsrp'])

        # Log to TensorBoard
        if writer:
            writer.add_scalar('Episode/Reward', metrics['reward'], episode)
            writer.add_scalar('Episode/Length', metrics['length'], episode)
            writer.add_scalar('Episode/Handovers', metrics['handovers'], episode)
            writer.add_scalar('Episode/AvgRSRP', metrics['avg_rsrp'], episode)
            writer.add_scalar('Episode/PingPongs', metrics['ping_pongs'], episode)
            writer.add_scalar('Training/Loss', metrics['loss'], episode)

            # Agent-specific metrics
            agent_config = agent.get_config()
            if 'epsilon' in agent_config:
                writer.add_scalar('Agent/Epsilon', agent_config['epsilon'], episode)
            if 'buffer_size' in agent_config:
                writer.add_scalar('Agent/BufferSize', agent_config['buffer_size'], episode)

        # Periodic logging
        log_interval = level_config['log_interval']
        if (episode + 1) % log_interval == 0:
            recent_rewards = episode_rewards[-log_interval:]
            recent_handovers = episode_handovers[-log_interval:]

            logger.info(
                f"Episode {episode + 1:4d}/{num_episodes}: "
                f"reward={np.mean(recent_rewards):+.2f}±{np.std(recent_rewards):.2f}, "
                f"handovers={np.mean(recent_handovers):.1f}±{np.std(recent_handovers):.1f}, "
                f"loss={metrics['loss']:.4f}"
            )

        # Save checkpoint
        checkpoint_interval = level_config['checkpoint_interval']
        if (episode + 1) % checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_ep{episode+1}.pth'
            agent.save(str(checkpoint_path))
            logger.info(f"✅ Checkpoint saved: {checkpoint_path}")

            # Save best model
            if metrics['reward'] > best_reward:
                best_reward = metrics['reward']
                best_path = checkpoint_dir / 'best_model.pth'
                agent.save(str(best_path))
                logger.info(f"🏆 New best model saved: {best_path}")

    # Save final model
    final_path = checkpoint_dir / 'final_model.pth'
    agent.save(str(final_path))
    logger.info(f"✅ Final model saved: {final_path}")

    # Close TensorBoard writer
    if writer:
        writer.close()

    # Training summary
    logger.info(f"\n{'='*80}")
    logger.info("Training Complete!")
    logger.info(f"{'='*80}")
    logger.info(f"Training Level: {args.level} ({level_config['name']})")
    logger.info(f"Algorithm: {args.algorithm.upper()}")
    logger.info(f"Total episodes: {num_episodes}")
    logger.info(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    logger.info(f"Best reward: {best_reward:.2f}")
    logger.info(f"Average handovers: {np.mean(episode_handovers):.1f} ± {np.std(episode_handovers):.1f}")
    logger.info(f"Average RSRP: {np.mean(episode_avg_rsrp):.1f} dBm")

    # Agent statistics
    agent_config = agent.get_config()
    if 'epsilon' in agent_config:
        logger.info(f"Final epsilon: {agent_config['epsilon']:.3f}")
    if 'buffer_size' in agent_config:
        logger.info(f"Buffer size: {agent_config['buffer_size']}")
    if 'training_steps' in agent_config:
        logger.info(f"Training steps: {agent_config['training_steps']}")

    logger.info(f"\nModels saved to: {checkpoint_dir}")
    logger.info(f"Logs saved to: {log_dir}")

    return {
        'episode_rewards': episode_rewards,
        'episode_handovers': episode_handovers,
        'episode_avg_rsrp': episode_avg_rsrp,
        'agent_config': agent_config,
    }


# ========== Main Entry Point ==========

def main():
    parser = argparse.ArgumentParser(
        description='Train RL agent for satellite handover optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick validation (Level 1: 2 hours, 20 satellites, 100 episodes)
    python train.py --algorithm dqn --level 1 --output-dir output/level1

    # Validation (Level 3: 10 hours, 101 satellites, 500 episodes)
    python train.py --algorithm dqn --level 3 --output-dir output/level3

    # Full training (Level 5: 35 hours, 101 satellites, 1700 episodes)
    python train.py --algorithm dqn --level 5 --output-dir output/level5

Multi-Level Training Strategy:
    Level 0: Smoke test (10 min, 10 satellites, 10 episodes)
    Level 1: Quick validation (2h, 20 satellites, 100 episodes) ⭐ Recommended
    Level 2: Development (6h, 50 satellites, 300 episodes)
    Level 3: Validation (10h, 101 satellites, 500 episodes)
    Level 4: Baseline (21h, 101 satellites, 1000 episodes)
    Level 5: Full training (35h, 101 satellites, 1700 episodes)
        """
    )

    # Algorithm selection
    parser.add_argument(
        '--algorithm', type=str,
        choices=list(ALGORITHM_REGISTRY.keys()),
        required=True,
        help='RL algorithm to use'
    )

    # Training level (P0 CRITICAL)
    parser.add_argument(
        '--level', type=int,
        choices=[0, 1, 2, 3, 4, 5],
        required=True,
        help='Training level (0=smoke test, 5=full training)'
    )

    # Configuration
    parser.add_argument(
        '--config', type=str,
        default='config/data_gen_config.yaml',
        help='Path to configuration file'
    )

    # Output
    parser.add_argument(
        '--output-dir', type=str,
        required=True,
        help='Output directory for checkpoints and logs'
    )

    # Reproducibility
    parser.add_argument(
        '--seed', type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    # Resume training
    parser.add_argument(
        '--resume', type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    args = parser.parse_args()

    # Load base config
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
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.995,
        }

    # Get training level config
    level_config = get_level_config(args.level)

    # Setup logging
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir / 'logs')

    # Set random seeds
    set_random_seeds(args.seed)
    logger.info(f"Random seed set to: {args.seed}")

    # Log configuration
    logger.info(f"\n{'='*80}")
    logger.info("Training Configuration")
    logger.info(f"{'='*80}")
    logger.info(f"Algorithm: {args.algorithm.upper()}")
    logger.info(f"Training Level: {args.level} ({level_config['name']})")
    logger.info(f"Estimated Time: {level_config['estimated_time_hours']:.1f}h")
    logger.info(f"Satellites: {level_config['num_satellites']}")
    logger.info(f"Episodes: {level_config['num_episodes']}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Config File: {args.config}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"{'='*80}\n")

    # Train
    results = train(config, level_config, args, logger)

    return results


if __name__ == '__main__':
    main()
