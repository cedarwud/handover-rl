"""
Orbit-Engine Adapters

Non-invasive integration with orbit-engine computational modules.
Reuses SGP4, ITU-R, and 3GPP calculations without modifying orbit-engine code.

Classes:
    OrbitEngineAdapter: Main adapter for calculating satellite states
    TLELoader: TLE data management (30-day precision strategy)
    TLE: Single TLE data structure
"""

from .orbit_engine_adapter import OrbitEngineAdapter
from .tle_loader import TLELoader, TLE
from .orbit_precompute_generator import OrbitPrecomputeGenerator
from .orbit_precompute_table import OrbitPrecomputeTable
from .adapter_wrapper import AdapterWrapper

__all__ = [
    'OrbitEngineAdapter',
    'TLELoader',
    'TLE',
    'OrbitPrecomputeGenerator',
    'OrbitPrecomputeTable',
    'AdapterWrapper',
]

__version__ = "3.0.0"  # Precompute system added
