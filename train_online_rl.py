#!/usr/bin/env python3
"""
Phase 3: Online RL Training Loop

Train DQN agent with multi-satellite environment using online RL

Academic Compliance:
- Online RL (agent explores environment)
- No pre-labeled data
- Real TLE data + complete physics
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
    print("⚠️  TensorBoard not available - metrics will not be logged")

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from adapters.orbit_engine_adapter import OrbitEngineAdapter
from environments.satellite_handover_env import SatelliteHandoverEnv
from agents.dqn_agent_v2 import DQNAgent
from utils.satellite_utils import load_stage4_optimized_satellites, verify_satellite_pool_integrity


def setup_logging(log_dir: Path):
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


def train(config, args):
    """
    Main training loop

    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    logger = logging.getLogger(__name__)

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

    # Satellite IDs - Load from orbit-engine Stage 4 Pool Optimization
    # NO HARDCODING: Uses scientifically selected satellites from six-stage processing
    # SOURCE: orbit-engine Stage 4 Pool Optimization output
    #
    # CONSTELLATION CHOICE: Starlink only (101 satellites)
    # REASON: Cross-constellation handover is NOT supported
    #         (Starlink and OneWeb are separate commercial networks)
    logger.info("=" * 80)
    logger.info("Loading optimized satellite pool from orbit-engine Stage 4...")
    logger.info("=" * 80)

    # Load Starlink-only pool (101 satellites)
    # Note: Cross-constellation handover between Starlink and OneWeb is not realistic
    satellite_ids, metadata = load_stage4_optimized_satellites(
        constellation_filter='starlink',
        return_metadata=True
    )

    # Verify integrity (pass metadata for accurate constellation detection)
    is_valid, message = verify_satellite_pool_integrity(
        satellite_ids,
        expected_constellation='starlink',
        metadata=metadata
    )
    logger.info(message)

    # If user requested subset for testing
    if args.num_satellites and args.num_satellites < len(satellite_ids):
        logger.info(f"\n⚠️  User requested subset: {args.num_satellites} satellites (for testing)")
        satellite_ids = satellite_ids[:args.num_satellites]
        logger.info(f"✅ Using first {len(satellite_ids)} satellites from optimized pool")

    logger.info(f"\n📊 Final Satellite Pool:")
    logger.info(f"   Constellation: Starlink")
    logger.info(f"   Total: {len(satellite_ids)} satellites")
    logger.info(f"   Source: orbit-engine Stage 4 Pool Optimization")
    logger.info(f"   Orbital Period: 95 minutes (550km altitude)")
    logger.info(f"   First 5: {satellite_ids[:5]}")
    logger.info(f"   Last 5: {satellite_ids[-5:]}")
    logger.info(f"=" * 80 + "\n")

    # Create environment
    logger.info("Creating environment...")
    env = SatelliteHandoverEnv(adapter, satellite_ids, config)
    logger.info("✅ Environment created")

    # Create agent
    logger.info("Creating DQN agent...")
    agent = DQNAgent(env.observation_space, env.action_space, config)
    logger.info("✅ Agent created")

    # Load checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        agent.load(args.resume)

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_handovers = []
    episode_avg_rsrp = []
    best_reward = -np.inf

    # ✅ Training Time Range: Continuous Time Sampling with Overlap
    # SCIENTIFIC BASIS:
    #   - Continuous time sampling (not random) - more systematic coverage
    #   - Sliding window with configurable overlap for data density
    #   - Multi-TLE strategy: 79 daily TLE files (2025-07-27 to 2025-10-17) = 82 days available
    #   - Orbital mechanics provides natural diversity (earth rotation)
    #
    # EPISODE STRUCTURE: Each episode = 95 minutes (1 complete Starlink orbital period)
    # OVERLAP: Configurable (0.0 = no overlap, 0.5 = 50% overlap)
    #   - No overlap: Episodes are back-to-back (episode_n+1 starts when episode_n ends)
    #   - 50% overlap: Episodes overlap by half (episode_n+1 starts at episode_n midpoint)
    #
    # TIME COVERAGE CALCULATION:
    #   - Without overlap: coverage = num_episodes × 95 min
    #   - With 50% overlap: coverage = num_episodes × 95 min × 0.5

    start_time_base = datetime(2025, 7, 27, 0, 0, 0)  # TLE data start date
    episode_duration_minutes = 95  # Starlink orbital period
    overlap_ratio = args.overlap  # Configurable overlap (default: 0.5)

    # Calculate time coverage
    if overlap_ratio > 0:
        episode_stride_minutes = episode_duration_minutes * (1 - overlap_ratio)
        total_coverage_minutes = args.num_episodes * episode_stride_minutes
    else:
        episode_stride_minutes = episode_duration_minutes
        total_coverage_minutes = args.num_episodes * episode_duration_minutes

    total_coverage_days = total_coverage_minutes / 60 / 24
    end_time = start_time_base + timedelta(minutes=total_coverage_minutes)

    # Calculate number of unique orbits (approximately)
    unique_orbits = int(total_coverage_minutes / episode_duration_minutes)

    logger.info(f"\n⏱️  Training Time Configuration:")
    logger.info(f"   Base time: {start_time_base.isoformat()}")
    logger.info(f"   End time: {end_time.isoformat()}")
    logger.info(f"   Time coverage: {total_coverage_days:.1f} days (~{unique_orbits} unique orbits)")
    logger.info(f"   Episode duration: {episode_duration_minutes} minutes (1 Starlink orbital period)")
    logger.info(f"   Episode overlap: {overlap_ratio * 100:.0f}%")
    logger.info(f"   Episode stride: {episode_stride_minutes:.1f} minutes")
    logger.info(f"   Sampling strategy: Continuous time (sliding window)")
    logger.info(f"   TLE precision: ±1 day (multi-TLE strategy)")
    logger.info(f"   Orbital diversity: Earth rotation provides {360 * total_coverage_days:.0f}° coverage")

    # Training loop
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting training: {args.num_episodes} episodes")
    logger.info(f"{'='*80}\n")

    for episode in tqdm(range(args.num_episodes), desc="Training"):
        # Continuous time sampling with sliding window
        # Each episode advances by episode_stride_minutes from the previous one
        # This provides systematic coverage of different orbital geometries
        time_offset_minutes = episode * episode_stride_minutes
        episode_start_time = start_time_base + timedelta(minutes=time_offset_minutes)

        # Reset environment
        obs, info = env.reset(
            seed=args.seed + episode,
            options={'start_time': episode_start_time}
        )

        episode_reward = 0
        episode_steps = 0
        episode_losses = []

        # Episode loop
        done = False
        while not done:
            # Select action
            action = agent.select_action(obs, training=True)

            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store experience
            agent.store_experience(obs, action, reward, next_obs, done)

            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            # Update metrics
            episode_reward += reward
            episode_steps += 1
            obs = next_obs

        # Update epsilon
        agent.update_epsilon()
        agent.episode_count += 1

        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        episode_handovers.append(info.get('num_handovers', 0))
        episode_avg_rsrp.append(info.get('avg_rsrp', 0))

        # Log to TensorBoard
        if writer:
            writer.add_scalar('Episode/Reward', episode_reward, episode)
            writer.add_scalar('Episode/Length', episode_steps, episode)
            writer.add_scalar('Episode/Handovers', info.get('num_handovers', 0), episode)
            writer.add_scalar('Episode/AvgRSRP', info.get('avg_rsrp', 0), episode)
            writer.add_scalar('Episode/PingPongs', info.get('num_ping_pongs', 0), episode)
            writer.add_scalar('Agent/Epsilon', agent.epsilon, episode)
            writer.add_scalar('Agent/BufferSize', len(agent.replay_buffer), episode)

            if episode_losses:
                writer.add_scalar('Training/Loss', np.mean(episode_losses), episode)

        # Periodic logging
        if (episode + 1) % args.log_interval == 0:
            recent_rewards = episode_rewards[-args.log_interval:]
            recent_handovers = episode_handovers[-args.log_interval:]

            logger.info(
                f"Episode {episode + 1:4d}/{args.num_episodes}: "
                f"reward={np.mean(recent_rewards):+.2f}±{np.std(recent_rewards):.2f}, "
                f"handovers={np.mean(recent_handovers):.1f}±{np.std(recent_handovers):.1f}, "
                f"buffer={len(agent.replay_buffer):5d}, "
                f"ε={agent.epsilon:.3f}"
            )

        # Save checkpoint
        if (episode + 1) % args.checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_ep{episode+1}.pth'
            agent.save(str(checkpoint_path))
            logger.info(f"✅ Checkpoint saved: {checkpoint_path}")

            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
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
    logger.info(f"Total episodes: {args.num_episodes}")
    logger.info(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    logger.info(f"Best reward: {best_reward:.2f}")
    logger.info(f"Average handovers: {np.mean(episode_handovers):.1f} ± {np.std(episode_handovers):.1f}")
    logger.info(f"Average RSRP: {np.mean(episode_avg_rsrp):.1f} dBm")
    logger.info(f"Final epsilon: {agent.epsilon:.3f}")
    logger.info(f"Buffer size: {len(agent.replay_buffer)}")
    logger.info(f"Training steps: {agent.training_steps}")
    logger.info(f"\nModels saved to: {checkpoint_dir}")
    logger.info(f"Logs saved to: {log_dir}")

    return {
        'episode_rewards': episode_rewards,
        'episode_handovers': episode_handovers,
        'episode_avg_rsrp': episode_avg_rsrp,
        'final_epsilon': agent.epsilon,
        'buffer_size': len(agent.replay_buffer),
    }


def main():
    parser = argparse.ArgumentParser(description='Train DQN agent for satellite handover')
    parser.add_argument('--config', type=str, default='config/data_gen_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--num-episodes', type=int, default=100,
                        help='Number of training episodes')
    parser.add_argument('--num-satellites', type=int, default=None,
                        help='Use subset of satellites for testing (default: None = use all 101 Starlink)')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Episode overlap ratio (0.0 = no overlap, 0.5 = 50%% overlap). Default: 0.5')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='output/training',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log every N episodes')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                        help='Save checkpoint every N episodes')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load config
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

    # Setup logging
    output_dir = Path(args.output_dir)
    setup_logging(output_dir / 'logs')

    # Train
    results = train(config, args)

    return results


if __name__ == '__main__':
    main()
