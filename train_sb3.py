#!/usr/bin/env python3
"""Train DQN using Stable Baselines3 for LEO Satellite Handover"""

import sys
import os
import logging
import argparse
from pathlib import Path
from datetime import datetime
import yaml
import numpy as np

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environments import SatelliteHandoverEnv
from adapters import AdapterWrapper
from utils.satellite_utils import load_stage4_optimized_satellites

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_env(config: dict, adapter, satellite_ids: list, log_dir: Path):
    """Create and wrap environment"""
    def _init():
        env = SatelliteHandoverEnv(adapter, satellite_ids, config)
        env = Monitor(env, str(log_dir / "monitor"))
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description='Train DQN with Stable Baselines3')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num-episodes', type=int, default=2500, help='Number of episodes')
    parser.add_argument('--eval-freq', type=int, default=50, help='Evaluation frequency (episodes)')
    parser.add_argument('--save-freq', type=int, default=100, help='Model save frequency (episodes)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set random seed if specified
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")

    # Load configuration
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # Initialize adapter
    logger.info("Initializing Orbit Adapter...")
    adapter = AdapterWrapper(config)

    # Load satellite pool
    logger.info("Loading satellite pool...")
    satellite_ids, metadata = load_stage4_optimized_satellites(
        constellation_filter='starlink',
        return_metadata=True,
        use_rl_training_data=False,
        use_candidate_pool=False
    )
    logger.info(f"Loaded {len(satellite_ids)} satellites from pool")

    # Create environment
    logger.info("Creating environment...")
    env = DummyVecEnv([make_env(config, adapter, satellite_ids, log_dir)])

    # Calculate timesteps from episodes
    # Each episode is ~10 minutes = 600 seconds / 5 seconds = 120 steps
    episode_duration_min = config.get('environment', {}).get('episode_duration_minutes', 10)
    time_step_seconds = config.get('environment', {}).get('time_step_seconds', 5)
    steps_per_episode = int((episode_duration_min * 60) / time_step_seconds)
    total_timesteps = args.num_episodes * steps_per_episode

    logger.info(f"Training settings:")
    logger.info(f"  Episodes: {args.num_episodes}")
    logger.info(f"  Steps per episode: {steps_per_episode}")
    logger.info(f"  Total timesteps: {total_timesteps}")

    # Extract agent config
    agent_config = config.get('agent', {})
    learning_rate = agent_config.get('learning_rate', 0.0001)
    gamma = agent_config.get('gamma', 0.95)
    batch_size = agent_config.get('batch_size', 64)
    buffer_capacity = agent_config.get('buffer_capacity', 10000)
    target_update_freq = agent_config.get('target_update_freq', 100)

    # SB3 uses exploration_fraction and exploration_final_eps instead of epsilon_decay
    epsilon_start = agent_config.get('epsilon_start', 0.82)
    epsilon_end = agent_config.get('epsilon_end', 0.2)

    # Calculate exploration_fraction from epsilon_decay
    # epsilon_decay of 0.9999 means we want slower decay
    # For 2500 episodes, let's use exploration_fraction = 0.8 (explore for 80% of training)
    exploration_fraction = 0.8 if agent_config.get('epsilon_decay', 0.9995) > 0.999 else 0.5

    logger.info(f"DQN Hyperparameters:")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Gamma: {gamma}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Buffer size: {buffer_capacity}")
    logger.info(f"  Target update interval: {target_update_freq}")
    logger.info(f"  Exploration: start={epsilon_start}, end={epsilon_end}, fraction={exploration_fraction}")

    # Create DQN model
    logger.info("Creating DQN model...")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_capacity,
        learning_starts=batch_size * 10,  # Start learning after 10 batches
        batch_size=batch_size,
        tau=1.0,  # Hard update
        gamma=gamma,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=target_update_freq,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=epsilon_start,
        exploration_final_eps=epsilon_end,
        policy_kwargs=dict(
            net_arch=[128, 128]  # Hidden dimensions from config
        ),
        tensorboard_log=str(log_dir),
        seed=args.seed,
        verbose=1,
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq * steps_per_episode,
        save_path=str(models_dir),
        name_prefix="dqn_model",
    )

    # Train
    logger.info(f"Starting training for {total_timesteps} timesteps...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback],
            log_interval=10,  # Log every 10 episodes
            progress_bar=True,
        )

        # Save final model
        final_model_path = models_dir / "dqn_final.zip"
        model.save(str(final_model_path))
        logger.info(f"✅ Training completed! Final model saved to {final_model_path}")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Evaluate final model
    logger.info("Evaluating final model...")
    eval_env = make_env(config, adapter, satellite_ids, log_dir)()

    total_episodes = 10
    episode_handovers = []
    episode_rewards = []

    for ep in range(total_episodes):
        obs, info = eval_env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated

        handovers = info.get('num_handovers', 0)
        episode_handovers.append(handovers)
        episode_rewards.append(episode_reward)
        logger.info(f"  Episode {ep+1}: {handovers} handovers, reward={episode_reward:.1f}")

    logger.info(f"\n📊 Evaluation Results:")
    logger.info(f"  Handovers: {np.mean(episode_handovers):.2f} ± {np.std(episode_handovers):.2f}")
    logger.info(f"  Reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
