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

__all__ = ['OrbitEngineAdapter', 'TLELoader', 'TLE']

__version__ = "2.0.0"
