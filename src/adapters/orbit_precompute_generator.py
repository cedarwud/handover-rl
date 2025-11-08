#!/usr/bin/env python3
"""
Orbit Precompute Generator - Generate Precomputed Orbit State Tables

用途: 預計算所有衛星在指定時間範圍內的軌道狀態
方法: 使用完整的 OrbitEngineAdapter (ITU-R + 3GPP + SGP4)
輸出: HDF5 格式的高效查詢表

Academic Standard:
- Uses complete physics models (no simplification)
- Real TLE data from Space-Track.org
- Generates reproducible state tables
- 100% traceable to orbit-engine calculations

Performance:
- Multiprocessing for parallel computation
- HDF5 with compression for efficient storage
- Progress bars (tqdm) for monitoring
- Estimated: ~30 min for 7 days × 125 satellites
"""

import h5py
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import logging
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

logger = logging.getLogger(__name__)


class OrbitPrecomputeGenerator:
    """
    Generate precomputed orbit state tables.

    使用完整的物理模型預計算軌道狀態，存儲為 HDF5 格式。
    訓練時使用 OrbitPrecomputeTable 進行 O(1) 查詢。

    Example:
        generator = OrbitPrecomputeGenerator(
            adapter=orbit_adapter,
            satellite_ids=all_satellite_ids,
            config=config
        )

        generator.generate(
            start_time=datetime(2025, 10, 7, 0, 0, 0),
            end_time=datetime(2025, 10, 14, 0, 0, 0),
            output_path="data/orbit_precompute_7days.h5",
            time_step_seconds=5
        )
    """

    # 12 state dimensions per satellite
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

    def __init__(self, adapter, satellite_ids: List[str], config: Dict):
        """
        Initialize generator.

        Args:
            adapter: OrbitEngineAdapter instance (with complete physics)
            satellite_ids: List of satellite IDs to precompute
            config: Configuration dictionary
        """
        self.adapter = adapter
        self.satellite_ids = satellite_ids
        self.config = config

        logger.info(f"OrbitPrecomputeGenerator initialized")
        logger.info(f"  Satellites: {len(satellite_ids)}")
        logger.info(f"  Adapter: {type(adapter).__name__}")

    def generate(self,
                 start_time: datetime,
                 end_time: datetime,
                 output_path: str,
                 time_step_seconds: int = 5,
                 num_processes: int = None):
        """
        Generate precomputed state table.

        Args:
            start_time: Start of time range (UTC)
            end_time: End of time range (UTC)
            output_path: Output HDF5 file path
            time_step_seconds: Time step (default: 5 seconds)
            num_processes: Number of parallel processes (default: CPU count - 1)

        Output HDF5 structure:
            /metadata/
                - generation_time
                - tle_epoch_start
                - tle_epoch_end
                - time_step_seconds
                - num_satellites
                - num_timesteps
                - satellite_ids[]

            /timestamps/
                - utc_timestamps[]  # Unix timestamps

            /states/
                - starlink_47925/
                    - rsrp_dbm[]
                    - rsrq_db[]
                    - ... (12 fields)
                - starlink_47926/
                    - ...
        """
        logger.info(f"Starting orbit state precomputation")
        logger.info(f"  Time range: {start_time} to {end_time}")
        logger.info(f"  Time step: {time_step_seconds} seconds")
        logger.info(f"  Satellites: {len(self.satellite_ids)}")

        # Generate timestamps
        timestamps = self._generate_timestamps(start_time, end_time, time_step_seconds)
        num_timesteps = len(timestamps)

        logger.info(f"  Total timesteps: {num_timesteps:,}")
        logger.info(f"  Estimated time: ~{self._estimate_time(num_timesteps)} minutes")

        # Determine number of processes
        if num_processes is None:
            num_processes = max(1, mp.cpu_count() - 1)

        logger.info(f"  Using {num_processes} parallel processes")

        # Create HDF5 file structure
        self._create_hdf5_structure(
            output_path,
            timestamps,
            start_time,
            end_time,
            time_step_seconds
        )

        # Compute states for each satellite (parallel)
        if num_processes > 1:
            self._compute_states_parallel(
                output_path,
                timestamps,
                num_processes
            )
        else:
            self._compute_states_serial(
                output_path,
                timestamps
            )

        # Validate and finalize
        self._validate_hdf5(output_path)

        logger.info(f"✅ Precomputation complete: {output_path}")
        logger.info(f"   File size: {self._get_file_size_mb(output_path):.1f} MB")

    def _generate_timestamps(self,
                            start_time: datetime,
                            end_time: datetime,
                            time_step_seconds: int) -> List[datetime]:
        """Generate list of timestamps."""
        timestamps = []
        current = start_time

        while current <= end_time:
            timestamps.append(current)
            current += timedelta(seconds=time_step_seconds)

        return timestamps

    def _create_hdf5_structure(self,
                               output_path: str,
                               timestamps: List[datetime],
                               start_time: datetime,
                               end_time: datetime,
                               time_step_seconds: int):
        """Create HDF5 file with proper structure."""
        num_timesteps = len(timestamps)

        with h5py.File(output_path, 'w') as f:
            # Metadata group
            meta = f.create_group('metadata')
            meta.attrs['generation_time'] = datetime.utcnow().isoformat()
            meta.attrs['tle_epoch_start'] = start_time.isoformat()
            meta.attrs['tle_epoch_end'] = end_time.isoformat()
            meta.attrs['time_step_seconds'] = time_step_seconds
            meta.attrs['num_satellites'] = len(self.satellite_ids)
            meta.attrs['num_timesteps'] = num_timesteps

            # Store satellite IDs as dataset
            satellite_ids_bytes = [sid.encode('utf-8') for sid in self.satellite_ids]
            meta.create_dataset(
                'satellite_ids',
                data=satellite_ids_bytes,
                dtype=h5py.string_dtype()
            )

            # Timestamps group
            ts_group = f.create_group('timestamps')

            # Convert to Unix timestamps (float64)
            unix_timestamps = np.array([
                ts.timestamp() for ts in timestamps
            ], dtype=np.float64)

            ts_group.create_dataset(
                'utc_timestamps',
                data=unix_timestamps,
                compression='gzip',
                compression_opts=4
            )

            # States group (will be filled by computation)
            states_group = f.create_group('states')

            # Pre-create datasets for each satellite
            for sat_id in self.satellite_ids:
                sat_group = states_group.create_group(sat_id)

                for field in self.STATE_FIELDS:
                    sat_group.create_dataset(
                        field,
                        shape=(num_timesteps,),
                        dtype=np.float32,
                        compression='gzip',
                        compression_opts=4,
                        fillvalue=np.nan  # Use NaN for missing/invalid states
                    )

        logger.info(f"HDF5 structure created: {output_path}")

    def _compute_states_serial(self,
                               output_path: str,
                               timestamps: List[datetime]):
        """Compute states serially (single process) with progress bar."""
        logger.info("Computing states (serial mode)...")

        with h5py.File(output_path, 'a') as f:
            states_group = f['states']

            # Progress bar for satellites
            for sat_id in tqdm(self.satellite_ids, desc="Satellites", unit="sat"):
                sat_group = states_group[sat_id]

                # Compute all timesteps for this satellite
                states_array = np.zeros((len(timestamps), len(self.STATE_FIELDS)), dtype=np.float32)

                for t_idx, timestamp in enumerate(timestamps):
                    try:
                        state_dict = self.adapter.calculate_state(
                            satellite_id=sat_id,
                            timestamp=timestamp
                        )

                        # Extract 12-dimensional state
                        for field_idx, field in enumerate(self.STATE_FIELDS):
                            states_array[t_idx, field_idx] = state_dict.get(field, np.nan)

                    except Exception as e:
                        # Fill with NaN on error
                        logger.debug(f"Error computing {sat_id} at {timestamp}: {e}")
                        states_array[t_idx, :] = np.nan

                # Write to HDF5
                for field_idx, field in enumerate(self.STATE_FIELDS):
                    sat_group[field][:] = states_array[:, field_idx]

    def _compute_states_parallel(self,
                                 output_path: str,
                                 timestamps: List[datetime],
                                 num_processes: int):
        """Compute states in parallel using multiprocessing."""
        logger.info(f"Computing states (parallel mode, {num_processes} processes)...")

        # Create worker function
        def compute_satellite_states(sat_id):
            """Worker function to compute all states for one satellite."""
            states_array = np.zeros((len(timestamps), len(self.STATE_FIELDS)), dtype=np.float32)

            for t_idx, timestamp in enumerate(timestamps):
                try:
                    state_dict = self.adapter.calculate_state(
                        satellite_id=sat_id,
                        timestamp=timestamp
                    )

                    for field_idx, field in enumerate(self.STATE_FIELDS):
                        states_array[t_idx, field_idx] = state_dict.get(field, np.nan)

                except Exception as e:
                    logger.debug(f"Error computing {sat_id} at {timestamp}: {e}")
                    states_array[t_idx, :] = np.nan

            return sat_id, states_array

        # Use multiprocessing pool
        with mp.Pool(num_processes) as pool:
            # Progress bar
            results = list(tqdm(
                pool.imap(compute_satellite_states, self.satellite_ids),
                total=len(self.satellite_ids),
                desc="Satellites",
                unit="sat"
            ))

        # Write results to HDF5
        logger.info("Writing results to HDF5...")
        with h5py.File(output_path, 'a') as f:
            states_group = f['states']

            for sat_id, states_array in tqdm(results, desc="Writing", unit="sat"):
                sat_group = states_group[sat_id]

                for field_idx, field in enumerate(self.STATE_FIELDS):
                    sat_group[field][:] = states_array[:, field_idx]

    def _validate_hdf5(self, output_path: str):
        """Validate HDF5 file integrity."""
        logger.info("Validating HDF5 file...")

        with h5py.File(output_path, 'r') as f:
            # Check metadata
            assert 'metadata' in f, "Missing metadata group"
            assert 'timestamps' in f, "Missing timestamps group"
            assert 'states' in f, "Missing states group"

            num_satellites = f['metadata'].attrs['num_satellites']
            num_timesteps = f['metadata'].attrs['num_timesteps']

            # Check timestamps
            timestamps = f['timestamps']['utc_timestamps'][:]
            assert len(timestamps) == num_timesteps, "Timestamp count mismatch"

            # Check states for each satellite
            states_group = f['states']
            assert len(states_group.keys()) == num_satellites, "Satellite count mismatch"

            for sat_id in states_group.keys():
                sat_group = states_group[sat_id]

                # Check all 12 fields exist
                for field in self.STATE_FIELDS:
                    assert field in sat_group, f"Missing field {field} for {sat_id}"
                    assert len(sat_group[field]) == num_timesteps, \
                        f"Timestep count mismatch for {sat_id}/{field}"

        logger.info("✅ HDF5 validation passed")

    def _estimate_time(self, num_timesteps: int) -> int:
        """Estimate computation time in minutes."""
        # Rough estimate: 0.5 seconds per (satellite, timestep)
        # With parallelization: divide by (num_processes)
        num_processes = max(1, mp.cpu_count() - 1)
        total_computations = len(self.satellite_ids) * num_timesteps
        estimated_seconds = total_computations * 0.5 / num_processes
        return int(estimated_seconds / 60)

    def _get_file_size_mb(self, path: str) -> float:
        """Get file size in MB."""
        import os
        return os.path.getsize(path) / 1024 / 1024


if __name__ == '__main__':
    # Example usage
    print("OrbitPrecomputeGenerator - Use scripts/generate_orbit_precompute.py instead")
