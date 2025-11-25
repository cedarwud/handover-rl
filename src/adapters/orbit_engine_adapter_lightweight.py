#!/usr/bin/env python3
"""
Lightweight OrbitEngineAdapter for multiprocessing workers.

Key differences from full OrbitEngineAdapter:
1. Accepts pre-loaded TLE data (no file I/O)
2. Minimal initialization overhead
3. Designed for parallel computation

This adapter is used by optimized precompute workers to avoid
repeated TLE file loading and heavy initialization.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add orbit-engine to path
ORBIT_ENGINE_ROOT = Path(__file__).parent.parent.parent.parent / "orbit-engine"
sys.path.insert(0, str(ORBIT_ENGINE_ROOT))

# Store current directory
_ORIGINAL_CWD = os.getcwd()

try:
    os.chdir(ORBIT_ENGINE_ROOT)
    from src.stages.stage2_orbital_computing.sgp4_calculator import SGP4Calculator
    from src.stages.stage5_signal_analysis.itur_physics_calculator import create_itur_physics_calculator
    from src.stages.stage5_signal_analysis.gpp_ts38214_signal_calculator import create_3gpp_signal_calculator
    from src.stages.stage5_signal_analysis.itur_official_atmospheric_model import create_itur_official_model
    os.chdir(_ORIGINAL_CWD)
except ImportError as e:
    os.chdir(_ORIGINAL_CWD)
    raise ImportError(f"Failed to import orbit-engine modules: {e}")


class TLE:
    """Simple TLE data class."""
    def __init__(self, line1: str, line2: str, epoch: datetime):
        self.line1 = line1
        self.line2 = line2
        self.epoch = epoch


class OrbitEngineAdapterLightweight:
    """
    Lightweight adapter for parallel workers.

    Differences from full OrbitEngineAdapter:
    - Accepts pre-loaded TLE data (no TLELoader initialization)
    - Minimal setup overhead
    - Designed for worker processes
    """

    def __init__(self, config: Dict, tle_data: Dict[str, Tuple[str, str, datetime]]):
        """
        Initialize lightweight adapter.

        Args:
            config: Configuration dictionary
            tle_data: Pre-loaded TLE data {sat_id: (line1, line2, epoch)}
        """
        self.config = config
        self.tle_data = tle_data

        # Ground station parameters
        self.ground_station = config['ground_station']
        self.lat = self.ground_station['latitude']
        self.lon = self.ground_station['longitude']
        self.alt_m = self.ground_station['altitude_m']

        # Physics parameters
        self.physics = config['physics']
        self.frequency_ghz = self.physics['frequency_ghz']
        self.bandwidth_mhz = self.physics['bandwidth_mhz']
        self.tx_power_dbm = self.physics['tx_power_dbm']

        # Initialize calculators
        self.sgp4_calc = SGP4Calculator()

        itur_config = {
            'frequency_ghz': self.frequency_ghz,
            **config.get('itur_physics', {})
        }
        self.itur_calc = create_itur_physics_calculator(itur_config)

        signal_calc_config = config['signal_calculator']
        self.gpp_calc = create_3gpp_signal_calculator(signal_calc_config)

        atmospheric_config = config['atmospheric_model']
        self.atmospheric_model = create_itur_official_model(
            temperature_k=atmospheric_config['temperature_k'],
            pressure_hpa=atmospheric_config['pressure_hpa'],
            water_vapor_density_g_m3=atmospheric_config['water_vapor_density_g_m3']
        )

    def calculate_state(self, satellite_id: str, timestamp: datetime,
                       tle: Optional[TLE] = None) -> Dict:
        """
        Calculate complete 12-dimensional state for a satellite.

        Args:
            satellite_id: NORAD catalog number
            timestamp: Calculation time (UTC)
            tle: TLE object (optional, will use pre-loaded if None)

        Returns:
            State dictionary with all 12 dimensions
        """
        # Get TLE from pre-loaded data
        if tle is None:
            if satellite_id not in self.tle_data:
                raise ValueError(f"No TLE data for satellite {satellite_id}")

            line1, line2, epoch = self.tle_data[satellite_id]
            tle = TLE(line1, line2, epoch)

        # Use SGP4 to propagate orbit
        sat_state = self.sgp4_calc.propagate(
            tle_line1=tle.line1,
            tle_line2=tle.line2,
            timestamp=timestamp
        )

        if sat_state is None:
            # Return NaN state if propagation fails
            return {field: np.nan for field in [
                'rsrp_dbm', 'rsrq_db', 'rs_sinr_db', 'distance_km',
                'elevation_deg', 'doppler_shift_hz', 'radial_velocity_ms',
                'atmospheric_loss_db', 'path_loss_db', 'propagation_delay_ms',
                'offset_mo_db', 'cell_offset_db'
            ]}

        # Calculate geometric parameters
        distance_km = sat_state['distance_km']
        elevation_deg = sat_state['elevation_deg']

        # Calculate path loss (ITU-R P.525)
        path_loss_db = self.itur_calc.calculate_free_space_path_loss(
            distance_km=distance_km,
            frequency_ghz=self.frequency_ghz
        )

        # Calculate atmospheric loss (ITU-R P.676-13)
        atmospheric_loss_db = self.atmospheric_model.calculate_atmospheric_loss(
            elevation_deg=elevation_deg,
            frequency_ghz=self.frequency_ghz
        )

        # Calculate total received power
        total_loss_db = path_loss_db + atmospheric_loss_db
        rx_power_dbm = self.tx_power_dbm - total_loss_db

        # Calculate 3GPP signal metrics (RSRP, RSRQ, SINR)
        signal_metrics = self.gpp_calc.calculate_signal_metrics(
            rx_power_dbm=rx_power_dbm,
            bandwidth_mhz=self.bandwidth_mhz
        )

        # Calculate Doppler shift
        radial_velocity_ms = sat_state.get('radial_velocity_ms', 0.0)
        doppler_shift_hz = (radial_velocity_ms / 299792.458) * (self.frequency_ghz * 1e9)

        # Calculate propagation delay
        propagation_delay_ms = (distance_km / 299.792458)

        # Compile complete state (12 dimensions)
        state = {
            # Signal Quality (3 dimensions)
            'rsrp_dbm': signal_metrics.get('rsrp_dbm', np.nan),
            'rsrq_db': signal_metrics.get('rsrq_db', np.nan),
            'rs_sinr_db': signal_metrics.get('rs_sinr_db', np.nan),

            # Physical Parameters (7 dimensions)
            'distance_km': distance_km,
            'elevation_deg': elevation_deg,
            'doppler_shift_hz': doppler_shift_hz,
            'radial_velocity_ms': radial_velocity_ms,
            'atmospheric_loss_db': atmospheric_loss_db,
            'path_loss_db': path_loss_db,
            'propagation_delay_ms': propagation_delay_ms,

            # 3GPP Offsets (2 dimensions)
            'offset_mo_db': signal_metrics.get('offset_mo_db', 0.0),
            'cell_offset_db': signal_metrics.get('cell_offset_db', 0.0),
        }

        return state
