#!/usr/bin/env python3
"""
Worker function for parallel precompute generation.

Must be at module level to be picklable for multiprocessing.
"""

import numpy as np
from typing import List, Tuple
from datetime import datetime


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


def compute_satellite_states(args: Tuple[str, dict, List[datetime]]) -> Tuple[str, np.ndarray]:
    """
    Worker function to compute states for one satellite across all timestamps.

    This function is at module level so it can be pickled for multiprocessing.

    Args:
        args: Tuple of (sat_id, config, timestamps_list)

    Returns:
        Tuple of (sat_id, states_array)
        states_array shape: (num_timesteps, 12)
    """
    sat_id, config, timestamps_list = args

    # Import here to avoid circular imports and ensure fresh imports in worker
    from adapters import OrbitEngineAdapter

    # Each worker creates its own adapter instance
    # This avoids multiprocessing serialization issues
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
            # Fill with NaN on error
            states_array[t_idx, :] = np.nan

    return sat_id, states_array
