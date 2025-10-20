#!/usr/bin/env python3
"""Quick check of what fields OrbitEngineAdapter actually returns"""

import sys
import yaml
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from adapters.orbit_engine_adapter import OrbitEngineAdapter
from utils.satellite_utils import load_satellite_ids

# Load config
with open('config/data_gen_config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize adapter
adapter = OrbitEngineAdapter(config)

# Test with a satellite - NO HARDCODING, extract from TLE
# SOURCE: Space-Track.org TLE data
satellite_ids = load_satellite_ids(max_satellites=1)
test_sat_id = satellite_ids[0]
print(f"Testing with satellite: {test_sat_id}")

test_time = datetime(2025, 10, 7, 12, 0, 0)
state = adapter.calculate_state(test_sat_id, test_time)

if state:
    print("Fields returned by OrbitEngineAdapter:")
    print("=" * 80)
    for key, value in sorted(state.items()):
        print(f"{key:30s} = {value}")
else:
    print("No state returned")
