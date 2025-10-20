#!/usr/bin/env python3
"""
Generate RL Training Dataset

Script to generate training/validation/test datasets using RLDataGenerator.

Usage:
    python scripts/generate_dataset.py --start-date 2024-01-01 --end-date 2024-01-07 --output data/episodes/train
"""

import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_generation import RLDataGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate RL training dataset")

    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for episodes"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/data_gen_config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--satellites",
        type=str,
        nargs="+",
        default=None,
        help="List of satellite IDs (default: use all from TLE)"
    )

    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    print("=" * 70)
    print("RL Dataset Generation")
    print("=" * 70)
    print(f"Start Date: {start_date.date()}")
    print(f"End Date: {end_date.date()}")
    print(f"Duration: {(end_date - start_date).days} days")
    print(f"Output: {args.output}")
    print(f"Config: {args.config}")
    print("=" * 70)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override satellites if specified
    if args.satellites:
        config['data_generation']['satellite_ids'] = args.satellites
        print(f"\nUsing specified satellites: {args.satellites}")

    # Initialize generator
    print("\nInitializing RLDataGenerator...")
    generator = RLDataGenerator(config)

    # Generate dataset
    print(f"\nGenerating episodes from {start_date.date()} to {end_date.date()}...")

    current_date = start_date
    episode_count = 0

    while current_date <= end_date:
        print(f"\nProcessing: {current_date.date()}")

        try:
            episodes = generator.generate_episodes_for_date(current_date)

            # Save episodes
            for episode in episodes:
                filename = f"episode_{episode['metadata']['episode_id']}.npz"
                filepath = output_dir / filename
                generator.save_episode(episode, str(filepath))
                episode_count += 1

            print(f"  ✅ Generated {len(episodes)} episodes")

        except Exception as e:
            print(f"  ❌ Error: {e}")

        current_date += timedelta(days=1)

    print("\n" + "=" * 70)
    print(f"✅ Dataset generation complete!")
    print(f"Total episodes generated: {episode_count}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
