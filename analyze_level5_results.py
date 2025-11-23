#!/usr/bin/env python3
"""
Analyze Level 5 Training Results
"""
import json
import glob
from pathlib import Path

def analyze_training():
    """Analyze Level 5 training results"""

    # Read progress file
    progress_file = Path("output/level5_full/training_progress.json")
    with open(progress_file, 'r') as f:
        progress = json.load(f)

    print("="*80)
    print("📊 Level 5 Training Results Summary")
    print("="*80)
    print()

    # Training overview
    print("🎯 Training Overview:")
    print(f"   Total Episodes: {progress['total_episodes']}")
    print(f"   Episodes per Batch: {progress['episodes_per_batch']}")
    print(f"   Total Batches: {progress['total_batches']}")
    print(f"   Completed Batches: {len(progress['completed_batches'])}")
    print(f"   Failed Batches: {len(progress['failed_batches'])}")
    print(f"   Success Rate: {len(progress['completed_batches'])/progress['total_batches']*100:.1f}%")
    print()

    # Time analysis
    from datetime import datetime
    start_time = datetime.fromisoformat(progress['start_time'])
    end_time = datetime.fromisoformat(progress['end_time'])
    duration = end_time - start_time
    duration_hours = duration.total_seconds() / 3600

    print("⏱️  Training Duration:")
    print(f"   Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Total Time: {duration_hours:.2f} hours ({duration_hours*60:.0f} minutes)")
    print(f"   Episodes/Hour: {progress['total_episodes']/duration_hours:.1f}")
    print(f"   Avg Time/Episode: {duration.total_seconds()/progress['total_episodes']:.1f} seconds")
    print()

    # Final checkpoint
    print("💾 Final Model:")
    print(f"   Checkpoint: {progress['final_checkpoint']}")

    checkpoint_path = Path(progress['final_checkpoint'])
    if checkpoint_path.exists():
        size_mb = checkpoint_path.stat().st_size / 1024 / 1024
        print(f"   Size: {size_mb:.2f} MB")
    print()

    # List all batch directories
    batch_dirs = sorted(glob.glob("output/level5_full/batch*"))
    print(f"📁 Batch Directories: {len(batch_dirs)} batches")
    print(f"   First: {Path(batch_dirs[0]).name}")
    print(f"   Last: {Path(batch_dirs[-1]).name}")
    print()

    print("="*80)
    print("✅ Level 5 Training Completed Successfully!")
    print("="*80)
    print()
    print("📈 Next Steps:")
    print("   1. Use TensorBoard to visualize learning curves:")
    print("      tensorboard --logdir output/level5_full")
    print()
    print("   2. Evaluate the trained model:")
    print("      python evaluate.py --checkpoint output/level5_full/batch16_ep1600-1700/checkpoints/final_model.pth")
    print()
    print("   3. Compare with RSRP baseline:")
    print("      python train.py --algorithm rsrp_baseline --level 5")
    print()

if __name__ == '__main__':
    analyze_training()
