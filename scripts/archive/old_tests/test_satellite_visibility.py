#!/usr/bin/env python3
"""
Test Satellite Visibility - Find visible satellite passes
"""

import sys
sys.path.insert(0, 'src')

from datetime import datetime, timedelta
import yaml

from adapters import OrbitEngineAdapter

print('=' * 70)
print('🔍 Testing Satellite Visibility')
print('=' * 70)

# Load configuration
with open('config/data_gen_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize adapter
print('🔧 Initializing OrbitEngineAdapter...')
adapter = OrbitEngineAdapter(config)

satellite_ids = config["data_generation"]["satellite_ids"]
print(f'✅ Testing satellites: {satellite_ids}')
print(f'   Ground Station: (24.9441°N, 121.3714°E)\n')

# Test visibility over 24 hours
start_date = datetime(2025, 10, 16, 0, 0, 0)
test_duration_hours = 24
time_step_minutes = 10  # Check every 10 minutes

print(f'📅 Test period: {start_date.strftime("%Y-%m-%d %H:%M")} + {test_duration_hours}h')
print(f'   Time step: {time_step_minutes} minutes\n')
print('=' * 70)

for sat_id in satellite_ids:
    print(f'\n🛰️  Satellite: {sat_id}')
    print('-' * 70)

    visible_passes = []
    current_time = start_date
    end_time = start_date + timedelta(hours=test_duration_hours)

    while current_time < end_time:
        try:
            state = adapter.calculate_state(
                satellite_id=sat_id,
                timestamp=current_time
            )

            if state and state.get('is_connectable', False):
                elevation = state.get('elevation_deg', 0)
                rsrp = state.get('rsrp_dbm', -999)
                distance_km = state.get('distance_km', 0)

                visible_passes.append({
                    'time': current_time,
                    'elevation': elevation,
                    'rsrp': rsrp,
                    'distance_km': distance_km
                })

                print(f'✅ {current_time.strftime("%H:%M")} - '
                      f'Elevation: {elevation:6.2f}° | '
                      f'RSRP: {rsrp:7.2f} dBm | '
                      f'Distance: {distance_km:7.1f} km')

        except Exception as e:
            print(f'❌ {current_time.strftime("%H:%M")} - Error: {e}')

        current_time += timedelta(minutes=time_step_minutes)

    print(f'\n📊 Summary for {sat_id}:')
    print(f'   Visible periods: {len(visible_passes)}/{int(test_duration_hours * 60 / time_step_minutes)}')
    print(f'   Visibility ratio: {len(visible_passes)/(test_duration_hours * 60 / time_step_minutes):.1%}')

    if visible_passes:
        elevations = [p['elevation'] for p in visible_passes]
        print(f'   Max elevation: {max(elevations):.2f}°')
        print(f'   Avg elevation: {sum(elevations)/len(elevations):.2f}°')
    else:
        print('   ⚠️  No visible passes found in this 24h period!')

print('\n' + '=' * 70)
print('🔍 Visibility Analysis Complete')
print('=' * 70)
