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

from ._precompute_worker import compute_satellite_states

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
        logger.info(f"  ⚠️  Note: Parallel mode requires OrbitEngineAdapter serialization")
        logger.info(f"     If parallel fails, will automatically fall back to serial mode")

    def _resolve_config_paths(self, config: Dict) -> Dict:
        """
        Resolve all relative paths in config to absolute paths.

        This is CRITICAL for multiprocessing to work correctly across orbit-engine updates.

        Problem: Multiprocessing子進程的工作目錄可能不同，相對路徑會解析失敗
        Solution: 轉換所有路徑為絕對路徑（在主進程中）

        Future-proof: 即使orbit-engine格式改變，只要config結構一致就能正常工作

        Args:
            config: Original config with potentially relative paths

        Returns:
            New config with all paths resolved to absolute paths
        """
        import copy
        from pathlib import Path
        import os

        # Deep copy to avoid modifying original config
        resolved_config = copy.deepcopy(config)

        # Get current working directory (main process)
        cwd = Path.cwd()

        # Resolve TLE paths in data_generation section
        if 'data_generation' in resolved_config:
            if 'tle_strategy' in resolved_config['data_generation']:
                tle_strategy = resolved_config['data_generation']['tle_strategy']
                if 'tle_directory' in tle_strategy:
                    tle_dir = tle_strategy['tle_directory']
                    # Convert to absolute path
                    abs_tle_dir = (cwd / tle_dir).resolve()
                    tle_strategy['tle_directory'] = str(abs_tle_dir)
                    logger.debug(f"Resolved TLE directory: {tle_dir} → {abs_tle_dir}")

        # Resolve paths in orbit_engine section
        if 'orbit_engine' in resolved_config:
            orbit_engine = resolved_config['orbit_engine']

            if 'orbit_engine_root' in orbit_engine:
                root = orbit_engine['orbit_engine_root']
                abs_root = (cwd / root).resolve()
                orbit_engine['orbit_engine_root'] = str(abs_root)
                logger.debug(f"Resolved orbit_engine_root: {root} → {abs_root}")

            if 'tle_data_dir' in orbit_engine:
                tle_data = orbit_engine['tle_data_dir']
                abs_tle_data = (cwd / tle_data).resolve()
                orbit_engine['tle_data_dir'] = str(abs_tle_data)
                logger.debug(f"Resolved tle_data_dir: {tle_data} → {abs_tle_data}")

        logger.info("✅ Config paths resolved to absolute paths for multiprocessing")
        return resolved_config

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
            # Use optimized parallel mode with pre-loaded TLE data
            self._compute_states_parallel_optimized(
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

            # No compression for timestamps (small dataset, fast access needed)
            ts_group.create_dataset(
                'utc_timestamps',
                data=unix_timestamps,
                compression=None
            )

            # States group (will be filled by computation)
            states_group = f.create_group('states')

            # Pre-create datasets for each satellite
            for sat_id in self.satellite_ids:
                sat_group = states_group.create_group(sat_id)

                for field in self.STATE_FIELDS:
                    # Optimized for training: No compression + chunk aligned to episode
                    # Episode = 20 min = 240 timesteps (at 5s intervals)
                    # This eliminates cross-chunk reads and decompression overhead
                    chunk_size = min(240, num_timesteps)  # Ensure chunk size doesn't exceed data size
                    sat_group.create_dataset(
                        field,
                        shape=(num_timesteps,),
                        dtype=np.float32,
                        compression=None,  # No compression for maximum speed
                        chunks=(chunk_size,) if num_timesteps > 1 else None,  # No chunking for single timestep
                        fillvalue=np.nan   # Use NaN for missing/invalid states
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

    def _preload_tle_data(self, reference_time: datetime) -> Dict[str, tuple]:
        """
        Preload TLE data from adapter to avoid repeated file I/O in workers.

        CRITICAL FIX: Use reference_time from precompute time range, not utcnow()!
        Previous bug: Used datetime.utcnow() which fails after orbit-engine updates
        when TLE files don't include current date.

        Args:
            reference_time: Time within the precompute range to use for TLE selection

        Returns:
            Dictionary mapping {sat_id: (line1, line2, epoch)}
        """
        logger.info(f"Preloading TLE data for parallel workers (reference: {reference_time.date()})...")
        tle_data = {}

        for sat_id in self.satellite_ids:
            try:
                # Get TLE from adapter's TLE loader for the actual precompute time range
                # NOT datetime.utcnow()! This ensures we get TLE data that matches
                # the time period we're computing states for.
                tle = self.adapter.tle_loader.get_tle_for_date(sat_id, reference_time)
                if tle:
                    tle_data[sat_id] = (tle.line1, tle.line2, tle.epoch)
                else:
                    logger.warning(f"No TLE found for {sat_id} at {reference_time.date()}")
            except Exception as e:
                logger.warning(f"Could not load TLE for {sat_id}: {e}")

        logger.info(f"  Preloaded TLE data for {len(tle_data)} satellites")
        return tle_data

    def _compute_states_parallel_optimized(self,
                                          output_path: str,
                                          timestamps: List[datetime],
                                          num_processes: int):
        """
        Optimized parallel computation with pre-loaded TLE data.

        This avoids the bottleneck of each worker reloading 230+ TLE files.

        If optimized parallel fails, will fall back to standard parallel or serial mode.
        """
        logger.info(f"✨ Using OPTIMIZED parallel mode ({num_processes} processes)...")
        logger.info("   Pre-loading TLE data to avoid repeated file I/O...")

        try:
            # Step 1: Preload TLE data in main process (done once)
            # CRITICAL FIX: Use first timestamp from precompute range, not utcnow()!
            tle_data = self._preload_tle_data(timestamps[0])

            if not tle_data:
                raise ValueError("No TLE data available for satellites")

            # Step 2: Import optimized worker
            from ._precompute_worker_optimized import compute_satellite_states_optimized

            # Step 3: Resolve config paths for multiprocessing
            # CRITICAL: Convert relative paths to absolute paths before passing to workers
            # This ensures paths work correctly regardless of worker process's CWD
            resolved_config = self._resolve_config_paths(self.config)

            # Step 4: Prepare arguments for each worker
            worker_args = [
                (sat_id, resolved_config, timestamps, tle_data)
                for sat_id in self.satellite_ids
            ]

            # Step 4: Run parallel computation
            with mp.Pool(num_processes) as pool:
                results = list(tqdm(
                    pool.imap(compute_satellite_states_optimized, worker_args),
                    total=len(self.satellite_ids),
                    desc="Satellites (optimized)",
                    unit="sat"
                ))

            logger.info("✅ Optimized parallel computation succeeded!")

            # Step 5: Write results to HDF5
            logger.info("Writing results to HDF5...")
            with h5py.File(output_path, 'a') as f:
                states_group = f['states']

                for sat_id, states_array in tqdm(results, desc="Writing", unit="sat"):
                    sat_group = states_group[sat_id]

                    for field_idx, field in enumerate(self.STATE_FIELDS):
                        sat_group[field][:] = states_array[:, field_idx]

        except Exception as e:
            logger.error(f"❌ Optimized parallel computation failed: {e}")
            logger.warning("Falling back to standard parallel mode...")
            # Fall back to standard parallel mode
            self._compute_states_parallel(output_path, timestamps, num_processes)

    def _compute_states_parallel(self,
                                 output_path: str,
                                 timestamps: List[datetime],
                                 num_processes: int):
        """
        Compute states in parallel using multiprocessing.

        ⚠️ WARNING: Multiprocessing with OrbitEngineAdapter may fail due to:
        - Complex object serialization issues
        - Shared state problems
        - orbit-engine internal state

        If parallel fails, will automatically fall back to serial mode.
        """
        logger.info(f"Attempting parallel computation ({num_processes} processes)...")
        logger.warning("⚠️  Parallel mode may fail with OrbitEngineAdapter")
        logger.warning("   Will automatically fall back to serial mode if it fails")

        try:
            # Use module-level worker function (must be picklable for multiprocessing)
            # See _precompute_worker.py for implementation

            # Resolve config paths for multiprocessing
            # CRITICAL: Convert relative paths to absolute paths before passing to workers
            resolved_config = self._resolve_config_paths(self.config)

            # Prepare arguments for each worker
            worker_args = [
                (sat_id, resolved_config, timestamps)
                for sat_id in self.satellite_ids
            ]

            # Use multiprocessing pool
            with mp.Pool(num_processes) as pool:
                # Progress bar
                results = list(tqdm(
                    pool.imap(compute_satellite_states, worker_args),
                    total=len(self.satellite_ids),
                    desc="Satellites",
                    unit="sat"
                ))

            logger.info("✅ Parallel computation succeeded")

        except Exception as e:
            logger.error(f"❌ Parallel computation failed: {e}")
            logger.warning("Falling back to serial mode...")
            # Fall back to serial computation
            self._compute_states_serial(output_path, timestamps)
            return

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
