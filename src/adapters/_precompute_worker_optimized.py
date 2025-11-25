#!/usr/bin/env python3
"""
Optimized worker function for parallel precompute generation.

Key optimizations:
1. Pre-loaded TLE data passed to workers (avoid repeated file I/O)
2. Minimal adapter initialization per worker
3. Reduced memory footprint

Must be at module level to be picklable for multiprocessing.
"""

import numpy as np
from typing import List, Tuple, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# State fields (must match OrbitPrecomputeGenerator.STATE_FIELDS)
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


def compute_satellite_states_optimized(args: Tuple[str, dict, List[datetime], dict]) -> Tuple[str, np.ndarray]:
    """
    Optimized worker function to compute states for one satellite.

    This function receives pre-loaded TLE data to avoid repeated file I/O.

    Args:
        args: Tuple of (sat_id, config, timestamps_list, tle_data_dict)
            - sat_id: Satellite ID
            - config: Configuration dictionary
            - timestamps_list: List of timestamps to compute
            - tle_data_dict: Pre-loaded TLE data {sat_id: (line1, line2, epoch)}

    Returns:
        Tuple of (sat_id, states_array)
        states_array shape: (num_timesteps, 12)
    """
    sat_id, config, timestamps_list, tle_data_dict = args

    # Import here to avoid circular imports
    from adapters.orbit_engine_adapter_lightweight import OrbitEngineAdapterLightweight

    try:
        # Create lightweight adapter with pre-loaded TLE data
        # This avoids re-loading 230 TLE files per worker
        worker_adapter = OrbitEngineAdapterLightweight(config, tle_data_dict)

        states_array = np.zeros((len(timestamps_list), len(STATE_FIELDS)), dtype=np.float32)

        for t_idx, timestamp in enumerate(timestamps_list):
            try:
                state_dict = worker_adapter.calculate_state(
                    satellite_id=sat_id,
                    timestamp=timestamp
                )

                for field_idx, field in enumerate(STATE_FIELDS):
                    states_array[t_idx, field_idx] = state_dict.get(field, np.nan)

            except Exception as e:
                # Log error for debugging (CRITICAL for future orbit-engine updates)
                logger.error(
                    f"Error computing state for satellite {sat_id} at {timestamp}: {e}",
                    exc_info=True  # Include full traceback for debugging
                )
                # Fill with NaN on error to allow processing to continue
                states_array[t_idx, :] = np.nan

        return sat_id, states_array

    except Exception as e:
        # If worker initialization fails, return NaN array
        logger.error(f"Worker failed for satellite {sat_id}: {e}")
        states_array = np.full((len(timestamps_list), len(STATE_FIELDS)), np.nan, dtype=np.float32)
        return sat_id, states_array


# Fallback to original worker if optimization fails
def compute_satellite_states(args: Tuple[str, dict, List[datetime]]) -> Tuple[str, np.ndarray]:
    """
    Original worker function (fallback).

    Args:
        args: Tuple of (sat_id, config, timestamps_list)

    Returns:
        Tuple of (sat_id, states_array)
    """
    sat_id, config, timestamps_list = args

    from adapters import OrbitEngineAdapter

    worker_adapter = OrbitEngineAdapter(config)
    states_array = np.zeros((len(timestamps_list), len(STATE_FIELDS)), dtype=np.float32)

    for t_idx, timestamp in enumerate(timestamps_list):
        try:
            state_dict = worker_adapter.calculate_state(
                satellite_id=sat_id,
                timestamp=timestamp
            )

            for field_idx, field in enumerate(STATE_FIELDS):
                states_array[t_idx, field_idx] = state_dict.get(field, np.nan)

        except Exception as e:
            # Log error for debugging (CRITICAL for future orbit-engine updates)
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                f"Error computing state for satellite {sat_id} at {timestamp}: {e}",
                exc_info=True  # Include full traceback for debugging
            )
            # Fill with NaN on error to allow processing to continue
            states_array[t_idx, :] = np.nan

    return sat_id, states_array
