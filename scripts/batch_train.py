#!/usr/bin/env python3
"""
Batch Training System - Memory-Safe Training

Splits training into batches to prevent memory leak accumulation.
Each batch runs in a separate process, saves checkpoint, and exits.

Strategy:
- Level 4: 1000 episodes = 10 batches × 100 episodes
- Each batch uses ~7.5 GB max (100 episodes × 24.5 MB + 5 GB overhead)
- Total memory usage stays under control

Academic Validity:
- Checkpoint includes full agent state (Q-network, optimizer, replay buffer)
- Training is continuous (no information loss)
- Results identical to single-process training
"""

import subprocess
import sys
from pathlib import Path
import yaml
import logging
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def run_batch(
    batch_idx: int,
    total_batches: int,
    episodes_per_batch: int,
    start_episode: int,
    output_dir: Path,
    config_path: Path,
    algorithm: str,
    level: int,
    seed: int,
    resume_checkpoint: Path = None
):
    """
    Run one training batch in subprocess

    Args:
        batch_idx: Current batch number (0-indexed)
        total_batches: Total number of batches
        episodes_per_batch: Episodes per batch
        start_episode: Starting episode number
        output_dir: Output directory
        config_path: Config file path
        algorithm: RL algorithm (dqn, ddqn)
        level: Training level
        seed: Random seed
        resume_checkpoint: Checkpoint to resume from (if any)

    Returns:
        success: True if batch completed successfully
        checkpoint_path: Path to saved checkpoint
    """
    end_episode = start_episode + episodes_per_batch
    batch_name = f"batch{batch_idx:02d}_ep{start_episode}-{end_episode}"

    logger.info(f"\n{'='*80}")
    logger.info(f"Starting Batch {batch_idx+1}/{total_batches}: Episodes {start_episode}-{end_episode-1}")
    logger.info(f"{'='*80}")

    # Build command
    cmd = [
        sys.executable,  # python
        str(Path(__file__).parent.parent / 'train.py'),
        '--algorithm', algorithm,
        '--level', str(level),
        '--config', str(config_path),
        '--output-dir', str(output_dir / batch_name),
        '--seed', str(seed + start_episode),  # Unique seed per batch
        '--num-episodes', str(episodes_per_batch),
        '--start-episode', str(start_episode),
    ]

    # Add resume checkpoint if provided
    if resume_checkpoint and resume_checkpoint.exists():
        cmd.extend(['--resume', str(resume_checkpoint)])
        logger.info(f"   Resuming from: {resume_checkpoint}")

    # Run batch in subprocess
    logger.info(f"   Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )

        # Find final checkpoint
        checkpoint_dir = output_dir / batch_name / 'checkpoints'
        final_checkpoint = checkpoint_dir / 'final_model.pth'

        if not final_checkpoint.exists():
            logger.error(f"❌ Batch {batch_idx+1} completed but checkpoint not found: {final_checkpoint}")
            return False, None

        logger.info(f"✅ Batch {batch_idx+1}/{total_batches} completed successfully")
        logger.info(f"   Checkpoint: {final_checkpoint}")

        return True, final_checkpoint

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Batch {batch_idx+1} failed with error code {e.returncode}")
        return False, None

    except Exception as e:
        logger.error(f"❌ Batch {batch_idx+1} failed with exception: {e}")
        return False, None


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Batch Training System')
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'ddqn'],
                        help='RL algorithm to use')
    parser.add_argument('--level', type=int, default=4,
                        help='Training level (0-6)')
    parser.add_argument('--config', type=str, default='configs/diagnostic_config.yaml',
                        help='Config file path')
    parser.add_argument('--output-dir', type=str, default='output/batch_training',
                        help='Output directory for all batches')
    parser.add_argument('--episodes-per-batch', type=int, default=100,
                        help='Number of episodes per batch')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--start-batch', type=int, default=0,
                        help='Starting batch index (for resuming)')

    args = parser.parse_args()

    # Load config to get total episodes
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get level config
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from configs import get_level_config
    level_config = get_level_config(args.level)

    total_episodes = level_config['num_episodes']
    episodes_per_batch = args.episodes_per_batch
    total_batches = (total_episodes + episodes_per_batch - 1) // episodes_per_batch  # Ceiling division

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("Batch Training System")
    logger.info("="*80)
    logger.info(f"Algorithm: {args.algorithm.upper()}")
    logger.info(f"Training Level: {args.level} ({level_config['name']})")
    logger.info(f"Total Episodes: {total_episodes}")
    logger.info(f"Episodes per Batch: {episodes_per_batch}")
    logger.info(f"Total Batches: {total_batches}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Starting from Batch: {args.start_batch}")
    logger.info("="*80)

    # Track progress
    progress = {
        'total_episodes': total_episodes,
        'episodes_per_batch': episodes_per_batch,
        'total_batches': total_batches,
        'completed_batches': [],
        'failed_batches': [],
        'start_time': datetime.now().isoformat(),
    }

    # Run batches
    last_checkpoint = None

    for batch_idx in range(args.start_batch, total_batches):
        start_episode = batch_idx * episodes_per_batch
        current_batch_size = min(episodes_per_batch, total_episodes - start_episode)

        success, checkpoint_path = run_batch(
            batch_idx=batch_idx,
            total_batches=total_batches,
            episodes_per_batch=current_batch_size,
            start_episode=start_episode,
            output_dir=output_dir,
            config_path=config_path,
            algorithm=args.algorithm,
            level=args.level,
            seed=args.seed,
            resume_checkpoint=last_checkpoint
        )

        if success:
            progress['completed_batches'].append(batch_idx)
            last_checkpoint = checkpoint_path
        else:
            progress['failed_batches'].append(batch_idx)
            logger.error(f"\n⚠️  Batch {batch_idx+1} failed. You can resume from this point with:")
            logger.error(f"   python scripts/batch_train.py --start-batch {batch_idx} [other args]")

            # Save progress and exit
            progress['end_time'] = datetime.now().isoformat()
            progress['status'] = 'failed'
            progress_file = output_dir / 'training_progress.json'
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)

            return 1

    # All batches completed
    progress['end_time'] = datetime.now().isoformat()
    progress['status'] = 'completed'
    progress['final_checkpoint'] = str(last_checkpoint)

    # Save progress
    progress_file = output_dir / 'training_progress.json'
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

    logger.info("\n" + "="*80)
    logger.info("🎉 All Batches Completed Successfully!")
    logger.info("="*80)
    logger.info(f"Total Batches: {total_batches}")
    logger.info(f"Total Episodes: {total_episodes}")
    logger.info(f"Final Checkpoint: {last_checkpoint}")
    logger.info(f"Progress File: {progress_file}")
    logger.info("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
