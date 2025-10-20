#!/usr/bin/env python3
"""
Universal RL Trainer - Algorithm-Agnostic Training Loop

Provides a universal training loop that works with any BaseRLAgent
and BaseHandoverEnvironment.

Features:
- Algorithm-agnostic training
- Checkpointing and model saving
- Validation and early stopping
- TensorBoard / WandB logging
- Progress tracking with tqdm

Supports:
- Online RL: DQN, PPO, SAC
- Offline RL: CQL, IQL
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm

# Optional dependencies
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .base_agent import BaseRLAgent
from .base_environment import BaseHandoverEnvironment


class UniversalRLTrainer:
    """
    Universal RL Trainer - Works with any algorithm.

    Usage:
        trainer = UniversalRLTrainer(agent, train_env, val_env, config)
        trainer.train()
    """

    def __init__(self,
                 agent: BaseRLAgent,
                 train_env: BaseHandoverEnvironment,
                 val_env: Optional[BaseHandoverEnvironment] = None,
                 config: Dict = None):
        """
        Initialize Universal Trainer.

        Args:
            agent: RL agent (must inherit from BaseRLAgent)
            train_env: Training environment
            val_env: Validation environment (optional)
            config: Training configuration
        """
        self.agent = agent
        self.train_env = train_env
        self.val_env = val_env
        self.config = config or {}

        # Training configuration
        training_config = self.config.get('training', {})
        self.max_episodes = training_config.get('epochs', 1000) * training_config.get('episodes_per_epoch', 1)
        self.max_steps = training_config.get('max_steps', None)

        # Checkpointing
        checkpoint_config = training_config.get('checkpoint', {})
        self.checkpoint_dir = Path(checkpoint_config.get('save_dir', 'checkpoints'))
        self.checkpoint_frequency = checkpoint_config.get('save_frequency', 100)
        self.save_best = checkpoint_config.get('save_best', True)
        self.best_metric = checkpoint_config.get('metric', 'avg_reward')
        self.best_value = -float('inf')

        # Validation
        validation_config = training_config.get('validation', {})
        self.val_frequency = validation_config.get('frequency', 50)
        self.val_episodes = validation_config.get('episodes', 10)

        # Logging
        logging_config = training_config.get('logging', {})
        self.log_frequency = logging_config.get('log_frequency', 10)
        self.use_tensorboard = logging_config.get('tensorboard', False) and TENSORBOARD_AVAILABLE
        self.use_wandb = logging_config.get('wandb', False) and WANDB_AVAILABLE

        # Initialize loggers
        if self.use_tensorboard:
            tb_dir = logging_config.get('tensorboard_dir', 'logs/tensorboard')
            self.writer = SummaryWriter(tb_dir)
            print(f"✅ TensorBoard logging enabled: {tb_dir}")

        if self.use_wandb:
            wandb_config = logging_config.get('wandb_config', {})
            wandb.init(
                project=wandb_config.get('project', 'handover-rl'),
                config=self.config
            )
            print(f"✅ WandB logging enabled")

        # Training state
        self.episode = 0
        self.total_steps = 0
        self.training_history = []

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        """
        Main training loop.

        Returns:
            training_history: List of episode statistics
        """
        print("=" * 70)
        print(f"Starting Training: {self.agent.__class__.__name__}")
        print("=" * 70)
        print(f"Max episodes: {self.max_episodes}")
        print(f"Agent: {self.agent}")
        print(f"Environment: {self.train_env.__class__.__name__}")
        print("=" * 70)

        start_time = time.time()

        # Training loop with progress bar
        pbar = tqdm(range(self.max_episodes), desc="Training")

        for episode in pbar:
            self.episode = episode

            # Train one episode
            episode_stats = self._train_episode()

            # Store history
            self.training_history.append(episode_stats)

            # Update progress bar
            pbar.set_postfix({
                'reward': f"{episode_stats['total_reward']:.2f}",
                'steps': episode_stats['steps'],
                'handovers': episode_stats['total_handovers']
            })

            # Logging
            if episode % self.log_frequency == 0:
                self._log_training(episode, episode_stats)

            # Validation
            if self.val_env and episode % self.val_frequency == 0 and episode > 0:
                val_stats = self._validate()
                self._log_validation(episode, val_stats)

                # Save best model
                if self.save_best:
                    metric_value = val_stats.get(self.best_metric, -float('inf'))
                    if metric_value > self.best_value:
                        self.best_value = metric_value
                        self._save_checkpoint('best.pth')
                        print(f"✅ New best model saved! {self.best_metric}={metric_value:.3f}")

            # Checkpointing
            if episode % self.checkpoint_frequency == 0 and episode > 0:
                self._save_checkpoint(f'episode_{episode}.pth')

            # Early stopping (if max steps reached)
            if self.max_steps and self.total_steps >= self.max_steps:
                print(f"\n⏹️  Training stopped: Max steps ({self.max_steps}) reached")
                break

        # Training complete
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"✅ Training Complete!")
        print(f"   Total episodes: {self.episode + 1}")
        print(f"   Total steps: {self.total_steps}")
        print(f"   Time elapsed: {elapsed/3600:.2f} hours")
        print(f"   Best {self.best_metric}: {self.best_value:.3f}")
        print("=" * 70)

        # Save final model
        self._save_checkpoint('final.pth')

        # Close loggers
        if self.use_tensorboard:
            self.writer.close()
        if self.use_wandb:
            wandb.finish()

        return self.training_history

    def _train_episode(self) -> Dict:
        """
        Train one episode.

        Returns:
            episode_stats: Dictionary with episode statistics
        """
        # Reset environment
        state, info = self.train_env.reset()

        # Episode state
        episode_reward = 0.0
        episode_steps = 0
        episode_losses = []

        # Notify agent of episode start
        self.agent.on_episode_start()

        # Episode loop
        while True:
            # Select action
            action = self.agent.select_action(state, eval_mode=False)

            # Execute action
            next_state, reward, terminated, truncated, info = self.train_env.step(action)

            # Update agent
            self.agent.on_step(state, action, reward, next_state, terminated or truncated)
            loss = self.agent.update()

            # Track statistics
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1

            if loss is not None:
                if isinstance(loss, dict):
                    episode_losses.append(loss.get('total_loss', 0.0))
                else:
                    episode_losses.append(loss)

            # Update state
            state = next_state

            # Check termination
            if terminated or truncated:
                break

        # Notify agent of episode end
        self.agent.on_episode_end(episode_reward, episode_steps)

        # Episode statistics
        episode_stats = {
            'episode': self.episode,
            'total_reward': episode_reward,
            'average_reward': episode_reward / episode_steps,
            'steps': episode_steps,
            'total_handovers': info.get('total_handovers', 0),
            'handover_rate': info.get('total_handovers', 0) / episode_steps,
            'average_loss': np.mean(episode_losses) if episode_losses else 0.0
        }

        return episode_stats

    def _validate(self) -> Dict:
        """
        Run validation on validation environment.

        Returns:
            val_stats: Aggregated validation statistics
        """
        if self.val_env is None:
            return {}

        # Set agent to evaluation mode
        self.agent.set_training_mode(False)

        # Run validation episodes
        all_rewards = []
        all_steps = []
        all_handovers = []

        for _ in range(self.val_episodes):
            state, info = self.val_env.reset()
            episode_reward = 0.0
            episode_steps = 0

            while True:
                action = self.agent.select_action(state, eval_mode=True)
                next_state, reward, terminated, truncated, info = self.val_env.step(action)

                episode_reward += reward
                episode_steps += 1
                state = next_state

                if terminated or truncated:
                    break

            all_rewards.append(episode_reward)
            all_steps.append(episode_steps)
            all_handovers.append(info.get('total_handovers', 0))

        # Set agent back to training mode
        self.agent.set_training_mode(True)

        # Aggregate statistics
        val_stats = {
            'avg_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'avg_steps': np.mean(all_steps),
            'avg_handovers': np.mean(all_handovers),
            'avg_handover_rate': np.mean([h/s for h, s in zip(all_handovers, all_steps)])
        }

        return val_stats

    def _log_training(self, episode: int, stats: Dict):
        """Log training statistics."""
        if self.use_tensorboard:
            self.writer.add_scalar('train/reward', stats['total_reward'], episode)
            self.writer.add_scalar('train/avg_reward', stats['average_reward'], episode)
            self.writer.add_scalar('train/steps', stats['steps'], episode)
            self.writer.add_scalar('train/handover_rate', stats['handover_rate'], episode)
            self.writer.add_scalar('train/loss', stats['average_loss'], episode)

        if self.use_wandb:
            wandb.log({
                'train/reward': stats['total_reward'],
                'train/avg_reward': stats['average_reward'],
                'train/steps': stats['steps'],
                'train/handover_rate': stats['handover_rate'],
                'train/loss': stats['average_loss'],
                'episode': episode
            })

    def _log_validation(self, episode: int, stats: Dict):
        """Log validation statistics."""
        if self.use_tensorboard:
            self.writer.add_scalar('val/avg_reward', stats['avg_reward'], episode)
            self.writer.add_scalar('val/std_reward', stats['std_reward'], episode)
            self.writer.add_scalar('val/avg_handover_rate', stats['avg_handover_rate'], episode)

        if self.use_wandb:
            wandb.log({
                'val/avg_reward': stats['avg_reward'],
                'val/std_reward': stats['std_reward'],
                'val/avg_handover_rate': stats['avg_handover_rate'],
                'episode': episode
            })

    def _save_checkpoint(self, filename: str):
        """
        Save training checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename

        # Save agent
        self.agent.save(str(checkpoint_path))

        print(f"💾 Checkpoint saved: {checkpoint_path}")

    def evaluate(self, num_episodes: int = 100) -> Dict:
        """
        Evaluate trained agent.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            eval_stats: Evaluation statistics
        """
        print(f"\n🧪 Evaluating agent ({num_episodes} episodes)...")

        self.agent.set_training_mode(False)

        all_rewards = []
        all_steps = []
        all_handovers = []

        for episode in tqdm(range(num_episodes), desc="Evaluation"):
            state, info = self.train_env.reset()
            episode_reward = 0.0
            episode_steps = 0

            while True:
                action = self.agent.select_action(state, eval_mode=True)
                next_state, reward, terminated, truncated, info = self.train_env.step(action)

                episode_reward += reward
                episode_steps += 1
                state = next_state

                if terminated or truncated:
                    break

            all_rewards.append(episode_reward)
            all_steps.append(episode_steps)
            all_handovers.append(info.get('total_handovers', 0))

        # Calculate statistics
        eval_stats = {
            'num_episodes': num_episodes,
            'avg_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'min_reward': np.min(all_rewards),
            'max_reward': np.max(all_rewards),
            'avg_steps': np.mean(all_steps),
            'avg_handovers': np.mean(all_handovers),
            'avg_handover_rate': np.mean([h/s for h, s in zip(all_handovers, all_steps)])
        }

        print(f"\n📊 Evaluation Results:")
        print(f"   Average Reward: {eval_stats['avg_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
        print(f"   Average Steps: {eval_stats['avg_steps']:.1f}")
        print(f"   Average Handover Rate: {eval_stats['avg_handover_rate']:.3f}")

        return eval_stats


# Example usage
if __name__ == "__main__":
    from .base_agent import DummyAgent
    from .base_environment import BaseHandoverEnvironment

    print("UniversalRLTrainer - Algorithm-Agnostic Training")
    print("=" * 60)

    # Configuration
    config = {
        'environment': {
            'state_dim': 12,
            'action_dim': 2,
            'reward_weights': {
                'qos_improvement': 1.0,
                'handover_penalty': 0.5,
                'signal_quality': 0.3,
                'ping_pong_penalty': 1.0
            },
            'max_steps_per_episode': 100
        },
        'training': {
            'epochs': 2,
            'episodes_per_epoch': 5,
            'checkpoint': {
                'save_frequency': 5,
                'save_best': True,
                'metric': 'avg_reward'
            },
            'validation': {
                'frequency': 5,
                'episodes': 3
            },
            'logging': {
                'log_frequency': 1,
                'tensorboard': False,
                'wandb': False
            }
        }
    }

    # Create agent and environments
    agent = DummyAgent(state_dim=12, action_dim=2, config={})
    train_env = BaseHandoverEnvironment(config)
    val_env = BaseHandoverEnvironment(config)

    # Create trainer
    trainer = UniversalRLTrainer(agent, train_env, val_env, config)

    # Train
    history = trainer.train()

    print(f"\n✅ Training complete! {len(history)} episodes")
    print(f"   Final reward: {history[-1]['total_reward']:.2f}")
