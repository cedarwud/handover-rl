#!/usr/bin/env python3
"""
Generate Orbit Precompute Table

用途: 生成預計算軌道狀態表，用於加速訓練
方法: 使用完整的 OrbitEngineAdapter 計算所有 (satellite, timestamp) 狀態
輸出: HDF5 格式，可用於快速查詢

Usage:
    # Generate 7-day table (recommended for Level 5 training)
    python scripts/generate_orbit_precompute.py \\
        --start-time "2025-10-07 00:00:00" \\
        --end-time "2025-10-14 00:00:00" \\
        --output data/orbit_precompute_7days.h5 \\
        --config config/diagnostic_config.yaml

    # Generate 14-day table (for longer experiments)
    python scripts/generate_orbit_precompute.py \\
        --start-time "2025-10-07 00:00:00" \\
        --end-time "2025-10-21 00:00:00" \\
        --output data/orbit_precompute_14days.h5 \\
        --config config/diagnostic_config.yaml \\
        --processes 16

Performance:
    - 7 days × 125 satellites × 5s timestep ≈ 2.1M computations
    - With 16 processes: ~20-30 minutes
    - Output size: ~700 MB (with compression)

Academic Standard:
    - Uses complete ITU-R P.676-13 + 3GPP TS 38.214/215
    - Real TLE data from Space-Track.org
    - No simplifications or approximations
    - 100% reproducible
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adapters import OrbitEngineAdapter, OrbitPrecomputeGenerator
from utils.satellite_utils import load_stage4_optimized_satellites

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate precomputed orbit state table',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--start-time',
        type=str,
        required=True,
        help='Start time (UTC) in format "YYYY-MM-DD HH:MM:SS"'
    )

    parser.add_argument(
        '--end-time',
        type=str,
        required=True,
        help='End time (UTC) in format "YYYY-MM-DD HH:MM:SS"'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output HDF5 file path (e.g., data/orbit_precompute_7days.h5)'
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Training config file (e.g., config/diagnostic_config.yaml)'
    )

    parser.add_argument(
        '--time-step',
        type=int,
        default=5,
        help='Time step in seconds (default: 5)'
    )

    parser.add_argument(
        '--processes',
        type=int,
        default=None,
        help='Number of parallel processes (default: CPU count - 1)'
    )

    parser.add_argument(
        '--satellite-limit',
        type=int,
        default=None,
        help='Limit number of satellites (for testing, default: all)'
    )

    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Auto-confirm (skip interactive prompt)'
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration file."""
    logger.info(f"Loading config: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_satellite_ids(limit=None):
    """Get list of satellite IDs using same logic as train.py."""
    logger.info("Loading satellite pool from orbit-engine Stage 4...")

    # Use same satellite loading as train.py
    satellite_ids, metadata = load_stage4_optimized_satellites(
        constellation_filter='starlink',
        return_metadata=True,
        use_rl_training_data=False,   # Use standard stage4 output path
        use_candidate_pool=False       # Use optimized pool (not candidate pool)
    )

    logger.info(f"  Total from optimized pool: {len(satellite_ids)}")

    if limit:
        satellite_ids = satellite_ids[:limit]
        logger.warning(f"  Limited to first {limit} satellites (for testing)")

    logger.info(f"  Will precompute: {len(satellite_ids)} satellites")

    return satellite_ids


def main():
    """Main function."""
    args = parse_args()

    # Parse timestamps
    try:
        start_time = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(args.end_time, "%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        logger.error(f"Invalid time format: {e}")
        logger.error('Use format: "YYYY-MM-DD HH:MM:SS"')
        sys.exit(1)

    # Validate time range
    if end_time <= start_time:
        logger.error("End time must be after start time")
        sys.exit(1)

    duration_days = (end_time - start_time).total_seconds() / 86400
    logger.info(f"\n{'='*60}")
    logger.info(f"Orbit Precompute Table Generation")
    logger.info(f"{'='*60}")
    logger.info(f"Start time:    {start_time}")
    logger.info(f"End time:      {end_time}")
    logger.info(f"Duration:      {duration_days:.1f} days")
    logger.info(f"Time step:     {args.time_step} seconds")
    logger.info(f"Output file:   {args.output}")
    logger.info(f"Config:        {args.config}")
    logger.info(f"{'='*60}\n")

    # Load config
    config = load_config(args.config)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize OrbitEngineAdapter (with complete physics)
    logger.info("Initializing OrbitEngineAdapter (complete physics)...")
    logger.info("  This uses ITU-R P.676-13 + 3GPP TS 38.214/215 + SGP4")

    adapter = OrbitEngineAdapter(config)

    # Get satellite IDs (using same pool as train.py)
    satellite_ids = get_satellite_ids(limit=args.satellite_limit)

    # Initialize generator
    logger.info("\nInitializing OrbitPrecomputeGenerator...")
    generator = OrbitPrecomputeGenerator(
        adapter=adapter,
        satellite_ids=satellite_ids,
        config=config
    )

    # Estimate output size
    num_timesteps = int((end_time - start_time).total_seconds() / args.time_step) + 1
    estimated_size_mb = (len(satellite_ids) * num_timesteps * 12 * 4) / (1024 * 1024)

    logger.info(f"\nEstimates:")
    logger.info(f"  Timesteps:     {num_timesteps:,}")
    logger.info(f"  Total states:  {len(satellite_ids) * num_timesteps:,}")
    logger.info(f"  Output size:   ~{estimated_size_mb:.1f} MB (with compression)")

    # Confirm with user
    print(f"\n{'='*60}")
    print("Ready to generate precompute table.")
    print(f"This will take approximately {generator._estimate_time(num_timesteps)} minutes.")
    print(f"{'='*60}\n")

    if not args.yes:
        response = input("Continue? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            logger.info("Cancelled by user")
            sys.exit(0)
    else:
        logger.info("Auto-confirmed with --yes flag")

    # Generate table
    logger.info("\n" + "="*60)
    logger.info("Starting generation...")
    logger.info("="*60 + "\n")

    try:
        generator.generate(
            start_time=start_time,
            end_time=end_time,
            output_path=str(output_path),
            time_step_seconds=args.time_step,
            num_processes=args.processes
        )

        logger.info("\n" + "="*60)
        logger.info("✅ Generation complete!")
        logger.info("="*60)
        logger.info(f"Output file: {output_path}")
        logger.info(f"File size:   {generator._get_file_size_mb(str(output_path)):.1f} MB")
        logger.info("\nNext steps:")
        logger.info("1. Update config/diagnostic_config.yaml:")
        logger.info("   precompute:")
        logger.info("     enabled: true")
        logger.info(f"     table_path: \"{output_path}\"")
        logger.info("\n2. Run training:")
        logger.info("   python train.py --config config/diagnostic_config.yaml")
        logger.info("="*60 + "\n")

    except KeyboardInterrupt:
        logger.warning("\n\nGeneration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
