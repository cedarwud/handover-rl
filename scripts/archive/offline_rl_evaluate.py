#!/usr/bin/env python3
"""
Evaluate Trained Model

Script to evaluate a trained DQN agent.

Usage:
    python scripts/evaluate_model.py --checkpoint checkpoints/dqn_model/best.pth --episodes data/episodes/test
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents import DQNAgent
from environments import HandoverEnvironment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained model")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--episodes",
        type=str,
        required=True,
        help="Path to test episode directory"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate"
    )

    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_args()

    print("=" * 70)
    print("Model Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test Episodes: {args.episodes}")
    print(f"Num Episodes: {args.num_episodes}")
    print("=" * 70)

    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize environment
    print("\nInitializing environment...")
    env = HandoverEnvironment(config)

    # Initialize agent
    print("Initializing agent...")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    agent.load(args.checkpoint)

    # Evaluate
    print("\n" + "=" * 70)
    print("Running Evaluation")
    print("=" * 70)

    episode_rewards = []
    episode_lengths = []

    for ep in range(args.num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Select action (greedy)
            action = agent.select_action(state, eval_mode=True)

            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1
            state = next_state

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if (ep + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {ep + 1}/{args.num_episodes} - Avg Reward (last 10): {avg_reward:.2f}")

    # Results
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Episodes evaluated: {args.num_episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
