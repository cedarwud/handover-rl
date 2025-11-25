#!/bin/bash
# Monitor precompute table generation progress

echo "======================================================"
echo "Precompute Generation Monitor"
echo "======================================================"
echo ""

# Check if process is running
MAIN_PID=$(ps aux | grep "generate_orbit_precompute.py" | grep "processes 16" | grep -v grep | head -1 | awk '{print $2}')

if [ -z "$MAIN_PID" ]; then
    echo "❌ Main process not found"
    exit 1
fi

echo "📊 Main Process: PID $MAIN_PID"
echo ""

# Count worker processes
WORKER_COUNT=$(ps aux | grep "generate_orbit_precompute.py" | grep -v grep | wc -l)
echo "👷 Worker Processes: $WORKER_COUNT (1 main + $(($WORKER_COUNT - 1)) workers)"
echo ""

# Check CPU usage
echo "💻 CPU Usage:"
ps aux | grep "generate_orbit_precompute.py" | grep -v grep | head -5 | awk '{print "   PID " $2 ": " $3 "% CPU, " $4 "% MEM"}'
echo ""

# Check if file exists and its size
if [ -f "data/orbit_precompute_30days_optimized.h5" ]; then
    FILE_SIZE=$(ls -lh data/orbit_precompute_30days_optimized.h5 | awk '{print $5}')
    echo "📁 Output File: $FILE_SIZE"

    # Try to read progress from HDF5
    source venv/bin/activate 2>/dev/null
    python -c "
import h5py
import numpy as np
try:
    with h5py.File('data/orbit_precompute_30days_optimized.h5', 'r') as f:
        if 'metadata/satellite_ids' in f:
            sat_ids = f['metadata/satellite_ids'][:]
            total_sats = len(sat_ids)

            if 'states' in f:
                completed = len([k for k in f['states'].keys() if isinstance(f['states'][k], h5py.Group)])
                print(f'   Progress: {completed}/{total_sats} satellites ({100*completed/total_sats:.1f}%)')

                # Check data validity
                if completed > 0:
                    first_sat = list(f['states'].keys())[0]
                    if 'elevation_deg' in f['states'][first_sat]:
                        elev = f['states'][first_sat]['elevation_deg'][:]
                        valid = np.sum(~np.isnan(elev))
                        print(f'   Data validity: {valid}/{len(elev)} values ({100*valid/len(elev):.1f}% valid)')
except:
    print('   File is being written, cannot read yet')
" 2>/dev/null
else
    echo "📁 Output File: Not created yet (initializing...)"
fi

echo ""
echo "======================================================"
