# Data Directory

This directory contains the precomputed orbit data required for training.

## Required Files

### Orbit Precompute Data
- **File**: `orbit_precompute_existing.h5`
- **Description**: 30-day precomputed satellite visibility and RSRP data
- **Size**: ~850 KB
- **Generation**: Run `python scripts/generate_orbit_precompute.py`

### Satellite IDs List
- **File**: `satellite_ids_from_precompute.txt`
- **Description**: List of 295 satellite IDs used for training
- **Format**: One satellite ID per line
- **Source**: Extracted from precompute data

## Directory Purpose

This directory is ignored by Git (except for this README and .gitkeep) to avoid committing large data files. When setting up a new environment:

1. Generate the precompute data:
   ```bash
   python scripts/generate_orbit_precompute.py
   ```

2. The satellite IDs list will be automatically extracted during generation

## Notes

- The precompute file covers 30 days of orbit data
- Data is indexed by 5-second intervals
- Contains visibility windows and RSRP values for all satellites
- Required for both training and evaluation
