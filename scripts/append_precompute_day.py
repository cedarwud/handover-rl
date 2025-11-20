#!/usr/bin/env python3
"""
Append 1 Day to Existing Precompute Table

追加 1 天數據到現有的 HDF5 預計算表，避免重新生成整個表。
"""

import h5py
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

# Add src to path (same as generate_orbit_precompute.py)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adapters import OrbitEngineAdapter
from tqdm import tqdm
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def append_one_day(
    input_hdf5: str,
    output_hdf5: str,
    adapter: OrbitEngineAdapter,
    prepend: bool = True
):
    """
    追加 1 天數據到現有的 HDF5 表。

    Args:
        input_hdf5: 現有的 HDF5 文件路徑
        output_hdf5: 輸出的 HDF5 文件路徑 (可以與輸入相同)
        adapter: OrbitEngineAdapter 實例
        prepend: True=往前追加 (10/09), False=往後追加 (11/09)
    """

    logger.info(f"\n{'='*60}")
    logger.info(f"Appending 1 day to precompute table")
    logger.info(f"{'='*60}")
    logger.info(f"Input:  {input_hdf5}")
    logger.info(f"Output: {output_hdf5}")
    logger.info(f"Mode:   {'Prepend (add earlier day)' if prepend else 'Append (add later day)'}")

    # Read existing file
    with h5py.File(input_hdf5, 'r') as f_in:
        # Get metadata
        start_time = datetime.fromisoformat(f_in['metadata'].attrs['tle_epoch_start'])
        end_time = datetime.fromisoformat(f_in['metadata'].attrs['tle_epoch_end'])
        time_step = int(f_in['metadata'].attrs['time_step_seconds'])  # Convert numpy.int64 to int
        num_satellites = int(f_in['metadata'].attrs['num_satellites'])
        satellite_ids = [sid.decode('utf-8') for sid in f_in['metadata']['satellite_ids'][:]]

        logger.info(f"\nCurrent table:")
        logger.info(f"  Time range: {start_time} to {end_time}")
        logger.info(f"  Days: {(end_time - start_time).days}")
        logger.info(f"  Time step: {time_step}s")
        logger.info(f"  Satellites: {num_satellites}")

        # Calculate new time range
        if prepend:
            new_start = start_time - timedelta(days=1)
            new_end = end_time
            logger.info(f"\nAdding 1 day before: {new_start.date()}")
        else:
            new_start = start_time
            new_end = end_time + timedelta(days=1)
            logger.info(f"\nAdding 1 day after: {new_end.date()}")

        total_days = (new_end - new_start).days
        logger.info(f"New time range: {new_start} to {new_end} ({total_days} days)")

        # Generate new timestamps for the additional day
        new_timestamps = []
        if prepend:
            current = new_start
            while current < start_time:
                new_timestamps.append(current)
                current += timedelta(seconds=time_step)
        else:
            current = end_time + timedelta(seconds=time_step)
            while current <= new_end:
                new_timestamps.append(current)
                current += timedelta(seconds=time_step)

        num_new_steps = len(new_timestamps)
        logger.info(f"New timesteps to compute: {num_new_steps:,}")

        # Read existing timestamps and states
        existing_unix_timestamps = f_in['timestamps']['utc_timestamps'][:]
        num_existing_steps = len(existing_unix_timestamps)

        # STATE FIELDS (must match OrbitPrecomputeGenerator.STATE_FIELDS)
        STATE_FIELDS = [
            'rsrp_dbm',
            'rsrq_db',
            'rs_sinr_db',
            'distance_km',
            'elevation_deg',
            'doppler_shift_hz',
            'radial_velocity_ms',
            'atmospheric_loss_db',
            'path_loss_db',
            'propagation_delay_ms',
            'offset_mo_db',
            'cell_offset_db',
        ]

        # Compute new states
        logger.info(f"\nComputing states for new timesteps...")
        new_states = {}

        for sat_id in tqdm(satellite_ids, desc="Satellites"):
            sat_states = {field: [] for field in STATE_FIELDS}

            for timestamp in new_timestamps:
                try:
                    state = adapter.get_satellite_state(sat_id, timestamp)

                    if state and state.get('is_visible', False):
                        for field in STATE_FIELDS:
                            sat_states[field].append(state.get(field, np.nan))
                    else:
                        # Not visible - fill with NaN
                        for field in STATE_FIELDS:
                            sat_states[field].append(np.nan)

                except Exception as e:
                    logger.warning(f"Error computing state for {sat_id} at {timestamp}: {e}")
                    for field in STATE_FIELDS:
                        sat_states[field].append(np.nan)

            # Convert to numpy arrays
            new_states[sat_id] = {
                field: np.array(values, dtype=np.float32)
                for field, values in sat_states.items()
            }

        logger.info(f"✅ Computed {num_new_steps:,} new timesteps for {num_satellites} satellites")

        # Create new HDF5 file with combined data
        logger.info(f"\nCreating new HDF5 file...")

        with h5py.File(output_hdf5, 'w') as f_out:
            # Metadata
            meta = f_out.create_group('metadata')
            meta.attrs['generation_time'] = datetime.utcnow().isoformat()
            meta.attrs['tle_epoch_start'] = new_start.isoformat()
            meta.attrs['tle_epoch_end'] = new_end.isoformat()
            meta.attrs['time_step_seconds'] = time_step
            meta.attrs['num_satellites'] = num_satellites
            meta.attrs['num_timesteps'] = num_existing_steps + num_new_steps

            satellite_ids_bytes = [sid.encode('utf-8') for sid in satellite_ids]
            meta.create_dataset(
                'satellite_ids',
                data=satellite_ids_bytes,
                dtype=h5py.string_dtype()
            )

            # Timestamps - combine old and new
            ts_group = f_out.create_group('timestamps')

            new_unix_timestamps = np.array([
                ts.timestamp() for ts in new_timestamps
            ], dtype=np.float64)

            if prepend:
                combined_timestamps = np.concatenate([new_unix_timestamps, existing_unix_timestamps])
            else:
                combined_timestamps = np.concatenate([existing_unix_timestamps, new_unix_timestamps])

            ts_group.create_dataset(
                'utc_timestamps',
                data=combined_timestamps,
                compression='gzip',
                compression_opts=4
            )

            # States - combine old and new for each satellite
            states_group = f_out.create_group('states')

            for sat_id in tqdm(satellite_ids, desc="Writing states"):
                sat_group = states_group.create_group(sat_id)

                for field in STATE_FIELDS:
                    # Read existing data
                    existing_data = f_in['states'][sat_id][field][:]

                    # Combine with new data
                    new_data = new_states[sat_id][field]

                    if prepend:
                        combined_data = np.concatenate([new_data, existing_data])
                    else:
                        combined_data = np.concatenate([existing_data, new_data])

                    sat_group.create_dataset(
                        field,
                        data=combined_data,
                        compression='gzip',
                        compression_opts=4
                    )

    logger.info(f"\n✅ Successfully created extended table: {output_hdf5}")
    logger.info(f"   New time range: {new_start} to {new_end}")
    logger.info(f"   Total days: {total_days}")
    logger.info(f"   Total timesteps: {num_existing_steps + num_new_steps:,}")

    # Get file size
    import os
    size_mb = os.path.getsize(output_hdf5) / (1024 * 1024)
    logger.info(f"   File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Append 1 day to precompute table")
    parser.add_argument("--input", required=True, help="Input HDF5 file")
    parser.add_argument("--output", required=True, help="Output HDF5 file")
    parser.add_argument("--config", default="config/diagnostic_config.yaml", help="Config file")
    parser.add_argument("--prepend", action="store_true", help="Add day BEFORE (default: add AFTER)")

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize adapter
    logger.info("Initializing OrbitEngineAdapter...")
    adapter = OrbitEngineAdapter(config)

    # Append day
    append_one_day(
        input_hdf5=args.input,
        output_hdf5=args.output,
        adapter=adapter,
        prepend=args.prepend
    )
