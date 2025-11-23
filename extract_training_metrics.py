#!/usr/bin/env python3
"""
Extract Training Metrics from TensorBoard Logs
"""
import glob
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import json

def extract_metrics(logdir):
    """Extract metrics from TensorBoard event files"""

    # Find all event files
    event_files = glob.glob(f"{logdir}/**/events.out.tfevents.*", recursive=True)

    print(f"📊 Level 5 Training Metrics Analysis")
    print("="*80)
    print(f"Found {len(event_files)} TensorBoard event files")
    print()

    all_rewards = []
    all_losses = []
    all_epsilons = []
    all_handovers = []
    all_rsrp = []

    for event_file in sorted(event_files):
        try:
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()

            # Get available tags
            tags = ea.Tags()

            # Extract scalars
            if 'Train/Reward' in tags['scalars']:
                rewards = ea.Scalars('Train/Reward')
                all_rewards.extend([(e.step, e.value) for e in rewards])

            if 'Train/Loss' in tags['scalars']:
                losses = ea.Scalars('Train/Loss')
                all_losses.extend([(e.step, e.value) for e in losses])

            if 'Train/Epsilon' in tags['scalars']:
                epsilons = ea.Scalars('Train/Epsilon')
                all_epsilons.extend([(e.step, e.value) for e in epsilons])

            if 'Handover/Count' in tags['scalars']:
                handovers = ea.Scalars('Handover/Count')
                all_handovers.extend([(e.step, e.value) for e in handovers])

            if 'Signal/Avg_RSRP' in tags['scalars']:
                rsrp = ea.Scalars('Signal/Avg_RSRP')
                all_rsrp.extend([(e.step, e.value) for e in rsrp])

        except Exception as e:
            print(f"Warning: Could not process {event_file}: {e}")
            continue

    # Sort by step (episode)
    all_rewards.sort(key=lambda x: x[0])
    all_losses.sort(key=lambda x: x[0])
    all_epsilons.sort(key=lambda x: x[0])
    all_handovers.sort(key=lambda x: x[0])
    all_rsrp.sort(key=lambda x: x[0])

    # Print summary
    print("📈 Training Summary:")
    print(f"   Total Episodes: {len(all_rewards)}")
    print()

    if all_rewards:
        rewards_values = [v for _, v in all_rewards]
        print("🎯 Reward Statistics:")
        print(f"   First 100 episodes: {np.mean(rewards_values[:100]):.2f} ± {np.std(rewards_values[:100]):.2f}")
        print(f"   Last 100 episodes:  {np.mean(rewards_values[-100:]):.2f} ± {np.std(rewards_values[-100:]):.2f}")
        print(f"   Overall mean:       {np.mean(rewards_values):.2f}")
        print(f"   Overall std:        {np.std(rewards_values):.2f}")
        print(f"   Min reward:         {np.min(rewards_values):.2f}")
        print(f"   Max reward:         {np.max(rewards_values):.2f}")
        print()

    if all_losses:
        loss_values = [v for _, v in all_losses if v > 0]  # Filter out zero losses
        if loss_values:
            print("📉 Loss Statistics:")
            print(f"   First 100 episodes: {np.mean(loss_values[:100]):.6f}")
            print(f"   Last 100 episodes:  {np.mean(loss_values[-100:]):.6f}")
            print(f"   Overall mean:       {np.mean(loss_values):.6f}")
            print()

    if all_epsilons:
        epsilon_values = [v for _, v in all_epsilons]
        print("🎲 Exploration (Epsilon):")
        print(f"   Initial:  {epsilon_values[0]:.4f}")
        print(f"   Final:    {epsilon_values[-1]:.4f}")
        print()

    if all_handovers:
        handover_values = [v for _, v in all_handovers]
        print("🔄 Handover Statistics:")
        print(f"   First 100 episodes: {np.mean(handover_values[:100]):.2f} handovers/episode")
        print(f"   Last 100 episodes:  {np.mean(handover_values[-100:]):.2f} handovers/episode")
        print(f"   Overall mean:       {np.mean(handover_values):.2f} handovers/episode")
        print()

    rsrp_values = []
    if all_rsrp:
        rsrp_values = [v for _, v in all_rsrp if v != 0]  # Filter out zero RSRP
        if rsrp_values:
            print("📡 Signal Quality (RSRP):")
            print(f"   First 100 episodes: {np.mean(rsrp_values[:100]):.2f} dBm")
            print(f"   Last 100 episodes:  {np.mean(rsrp_values[-100:]):.2f} dBm")
            print(f"   Overall mean:       {np.mean(rsrp_values):.2f} dBm")
            print()

    # Save detailed data
    output_file = Path(logdir).parent / "training_metrics_summary.json"
    summary = {
        "total_episodes": len(all_rewards),
        "rewards": {
            "first_100_mean": float(np.mean(rewards_values[:100])) if all_rewards else None,
            "last_100_mean": float(np.mean(rewards_values[-100:])) if all_rewards else None,
            "overall_mean": float(np.mean(rewards_values)) if all_rewards else None,
            "overall_std": float(np.std(rewards_values)) if all_rewards else None,
        },
        "handovers": {
            "first_100_mean": float(np.mean(handover_values[:100])) if all_handovers else None,
            "last_100_mean": float(np.mean(handover_values[-100:])) if all_handovers else None,
            "overall_mean": float(np.mean(handover_values)) if all_handovers else None,
        },
        "rsrp": {
            "first_100_mean": float(np.mean(rsrp_values[:100])) if rsrp_values else None,
            "last_100_mean": float(np.mean(rsrp_values[-100:])) if rsrp_values else None,
            "overall_mean": float(np.mean(rsrp_values)) if rsrp_values else None,
        }
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"💾 Detailed metrics saved to: {output_file}")
    print()
    print("="*80)
    print("✅ Analysis Complete!")
    print()
    print("🔗 View learning curves in TensorBoard:")
    print("   tensorboard --logdir output/level5_full")
    print(f"   Then open: http://localhost:6006")
    print()

if __name__ == '__main__':
    extract_metrics("output/level5_full")
