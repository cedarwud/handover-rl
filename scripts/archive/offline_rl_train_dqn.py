#!/usr/bin/env python3
"""
Train DQN Agent

Script to train DQN agent using UniversalRLTrainer.

Usage:
    python scripts/train_dqn.py --config config/training_config.yaml --episodes data/episodes/train
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_core import UniversalRLTrainer
from agents import DQNAgent
from environments import HandoverEnvironment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DQN agent")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file"
    )

    parser.add_argument(
        "--episodes",
        type=str,
        required=True,
        help="Path to episode directory"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/dqn_model",
        help="Output directory for checkpoints"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training"
    )

    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_args()

    print("=" * 70)
    print("DQN Agent Training")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Episodes: {args.episodes}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    if args.resume:
        print(f"Resume from: {args.resume}")
    print("=" * 70)

    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override device if specified
    if args.device != "auto":
        config['device'] = args.device

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize environment
    print("\nInitializing HandoverEnvironment...")
    env = HandoverEnvironment(config)

    # Initialize agent
    print("Initializing DQNAgent...")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nLoading checkpoint: {args.resume}")
        agent.load(args.resume)

    # Initialize trainer
    print("\nInitializing UniversalRLTrainer...")
    trainer = UniversalRLTrainer(
        agent=agent,
        env=env,
        config=config
    )

    # Load episodes
    print(f"\nLoading episodes from: {args.episodes}")
    episode_dir = Path(args.episodes)
    episode_files = list(episode_dir.glob("episode_*.npz"))
    print(f"Found {len(episode_files)} episodes")

    # Train
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    try:
        history = trainer.train(
            num_episodes=config.get('num_episodes', 1000),
            save_dir=str(output_dir),
            save_frequency=config.get('save_frequency', 100),
            eval_frequency=config.get('eval_frequency', 50)
        )

        print("\n" + "=" * 70)
        print("✅ Training Complete!")
        print("=" * 70)
        print(f"Total episodes: {len(history['episode_rewards'])}")
        print(f"Final average reward: {history['episode_rewards'][-100:].mean():.2f}")
        print(f"Best checkpoint saved to: {output_dir / 'best.pth'}")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("⚠️  Training interrupted by user")
        print("=" * 70)

        # Save final checkpoint
        final_path = output_dir / "interrupted.pth"
        agent.save(str(final_path))
        print(f"Checkpoint saved to: {final_path}")


if __name__ == "__main__":
    main()
