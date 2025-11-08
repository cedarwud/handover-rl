#!/usr/bin/env python3
"""
Adapter Wrapper - Unified Interface for Precompute and Real-time Calculation

用途: 統一的 Adapter 接口，根據配置自動選擇：
      - OrbitPrecomputeTable (快速查表)
      - OrbitEngineAdapter (實時計算)

優勢: 對訓練代碼完全透明，無需修改 Environment 代碼

Academic Standard:
- Both backends use same physics models
- Transparent switching via configuration
- No accuracy loss with precompute mode
- Backward compatible with existing code

Example:
    # In train.py or environment initialization
    from adapters import AdapterWrapper

    # Old way (direct adapter):
    # adapter = OrbitEngineAdapter(config)

    # New way (automatic selection):
    adapter = AdapterWrapper(config)

    # Use as before - no code changes needed!
    state = adapter.calculate_state(satellite_id, timestamp)

Configuration:
    # config.yaml
    precompute:
      enabled: true  # Use precompute table
      table_path: "data/orbit_precompute_7days.h5"

    # If enabled=false, automatically falls back to OrbitEngineAdapter
"""

from typing import Dict, List, Optional
from datetime import datetime
import logging
from pathlib import Path

from .orbit_precompute_table import OrbitPrecomputeTable
from .orbit_engine_adapter import OrbitEngineAdapter

logger = logging.getLogger(__name__)


class AdapterWrapper:
    """
    Unified adapter interface with automatic backend selection.

    自動根據配置選擇 backend:
    - config['precompute']['enabled'] = True  → OrbitPrecomputeTable
    - config['precompute']['enabled'] = False → OrbitEngineAdapter

    提供與 OrbitEngineAdapter 完全相同的 API，對使用者透明。
    """

    def __init__(self, config: Dict):
        """
        Initialize adapter wrapper.

        Args:
            config: Configuration dictionary with optional 'precompute' section:
                    {
                        'precompute': {
                            'enabled': bool,
                            'table_path': str
                        },
                        ... (other config for OrbitEngineAdapter)
                    }
        """
        self.config = config
        self.backend = None
        self.backend_type = None

        # Check if precompute is enabled
        precompute_config = config.get('precompute', {})
        use_precompute = precompute_config.get('enabled', False)

        if use_precompute:
            # Use precompute table
            table_path = precompute_config.get('table_path')

            if not table_path:
                logger.warning(
                    "Precompute enabled but no table_path specified. "
                    "Falling back to real-time calculation."
                )
                self._init_realtime_backend()
            else:
                # Check if file exists
                if not Path(table_path).exists():
                    logger.warning(
                        f"Precompute table not found: {table_path}. "
                        f"Falling back to real-time calculation."
                    )
                    self._init_realtime_backend()
                else:
                    self._init_precompute_backend(table_path)
        else:
            # Use real-time calculation
            self._init_realtime_backend()

        logger.info(f"AdapterWrapper initialized with backend: {self.backend_type}")

    def _init_precompute_backend(self, table_path: str):
        """Initialize precompute table backend."""
        logger.info(f"Loading precompute table: {table_path}")

        try:
            self.backend = OrbitPrecomputeTable(table_path)
            self.backend_type = "OrbitPrecomputeTable"

            # Log table info
            metadata = self.backend.get_metadata()
            logger.info(f"  Time range: {metadata['tle_epoch_start']} to {metadata['tle_epoch_end']}")
            logger.info(f"  Satellites: {metadata['num_satellites']}")
            logger.info(f"  Timesteps: {metadata['num_timesteps']:,}")
            logger.info("✅ Precompute mode enabled - Training will be ~100x faster!")

        except Exception as e:
            logger.error(f"Failed to load precompute table: {e}")
            logger.warning("Falling back to real-time calculation")
            self._init_realtime_backend()

    def _init_realtime_backend(self):
        """Initialize real-time calculation backend."""
        logger.info("Initializing real-time physics calculation backend")

        self.backend = OrbitEngineAdapter(self.config)
        self.backend_type = "OrbitEngineAdapter"

        logger.info("✅ Real-time calculation mode enabled")
        logger.info("⚠️  Training will be slow. Consider generating precompute table.")

    def calculate_state(self,
                       satellite_id: str,
                       timestamp: datetime,
                       tle: Optional[object] = None) -> Dict:
        """
        Calculate/query satellite state (unified API).

        This method provides the same interface as OrbitEngineAdapter.calculate_state()
        but automatically uses the configured backend (precompute or real-time).

        Args:
            satellite_id: Satellite ID
            timestamp: Query timestamp (UTC)
            tle: TLE object (only used in real-time mode, ignored in precompute mode)

        Returns:
            State dictionary with 12 fields + metadata
        """
        return self.backend.calculate_state(
            satellite_id=satellite_id,
            timestamp=timestamp,
            tle=tle
        )

    def get_available_satellites(self) -> List[str]:
        """
        Get list of available satellites.

        Unified interface for both backends.
        """
        if hasattr(self.backend, 'get_available_satellites'):
            return self.backend.get_available_satellites()
        else:
            # Fallback for OrbitEngineAdapter (doesn't have this method)
            # Return empty list or try to get from TLE loader
            if hasattr(self.backend, 'tle_loader'):
                return self.backend.tle_loader.get_available_satellites()
            else:
                logger.warning("Backend does not support get_available_satellites()")
                return []

    def get_backend_info(self) -> Dict:
        """
        Get information about current backend.

        Returns:
            {
                'backend_type': str,  # "OrbitPrecomputeTable" or "OrbitEngineAdapter"
                'is_precompute': bool,
                'metadata': dict      # Backend-specific metadata
            }
        """
        info = {
            'backend_type': self.backend_type,
            'is_precompute': self.backend_type == "OrbitPrecomputeTable",
        }

        # Add backend-specific metadata
        if hasattr(self.backend, 'get_metadata'):
            info['metadata'] = self.backend.get_metadata()
        else:
            info['metadata'] = {}

        return info

    def close(self):
        """Close backend resources."""
        if hasattr(self.backend, 'close'):
            self.backend.close()

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


if __name__ == '__main__':
    # Example usage
    import yaml
    from datetime import timedelta

    print("=== AdapterWrapper Test ===\n")

    # Test 1: Precompute mode (if table exists)
    print("Test 1: Precompute mode")
    config_precompute = {
        'precompute': {
            'enabled': True,
            'table_path': 'data/orbit_precompute_7days.h5'
        }
    }

    adapter_pre = AdapterWrapper(config_precompute)
    print(f"  Backend: {adapter_pre.backend_type}")
    print(f"  Info: {adapter_pre.get_backend_info()}\n")

    # Test 2: Real-time mode
    print("Test 2: Real-time mode")
    config_realtime = {
        'precompute': {
            'enabled': False
        },
        'ground_station': {
            'latitude_deg': 25.0,
            'longitude_deg': 121.5,
            'altitude_m': 50.0
        }
    }

    try:
        adapter_rt = AdapterWrapper(config_realtime)
        print(f"  Backend: {adapter_rt.backend_type}")
        print(f"  Info: {adapter_rt.get_backend_info()}\n")
    except Exception as e:
        print(f"  Error: {e}\n")

    # Cleanup
    adapter_pre.close()
    # adapter_rt.close()

    print("✅ AdapterWrapper test complete")
