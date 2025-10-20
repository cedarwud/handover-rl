#!/usr/bin/env python3
"""
TLE Loader - TLE Data Management

Manages loading and organizing TLE (Two-Line Element) files for satellite orbit propagation.

Design:
- Supports single TLE or multiple TLE files (for 30-day precision strategy)
- Validates TLE format and epoch dates
- Organizes TLEs by satellite NORAD ID
- Implements 30 TLE × 1 day strategy for <1km accuracy

SOURCE:
- TLE format: https://celestrak.org/NORAD/documentation/tle-fmt.php
- SGP4 precision: <1km within ±1-2 days from epoch
- Strategy: Use 30 TLE files, each propagated for 1 day
"""

import os
import glob
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import re


class TLE:
    """
    Single TLE (Two-Line Element) data structure.

    Attributes:
        satellite_id: NORAD catalog number
        satellite_name: Satellite name (from line 0)
        line1: TLE line 1
        line2: TLE line 2
        epoch: Epoch datetime
        file_path: Source file path
    """

    def __init__(self, satellite_id: str, satellite_name: str,
                 line1: str, line2: str, file_path: str = None):
        self.satellite_id = satellite_id
        self.satellite_name = satellite_name
        self.line1 = line1
        self.line2 = line2
        self.file_path = file_path
        self.epoch = self._parse_epoch(line1)

    def _parse_epoch(self, line1: str) -> datetime:
        """
        Parse epoch from TLE line 1.

        Format: YYDDDddddd (2-digit year, 3-digit day of year, fractional day)
        Position: Columns 19-32

        SOURCE: https://celestrak.org/NORAD/documentation/tle-fmt.php
        """
        try:
            # Extract epoch string (columns 19-32)
            epoch_str = line1[18:32].strip()

            # Parse year (2-digit)
            year_str = epoch_str[:2]
            year = int(year_str)
            # Convert 2-digit year to 4-digit (00-56 = 2000-2056, 57-99 = 1957-1999)
            year = 2000 + year if year < 57 else 1900 + year

            # Parse day of year with fractional part
            day_of_year = float(epoch_str[2:])

            # Calculate datetime
            epoch = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)

            return epoch

        except Exception as e:
            raise ValueError(f"Failed to parse TLE epoch from line 1: {e}")

    def is_valid_for_date(self, target_date: datetime, max_days: int = 2) -> bool:
        """
        Check if TLE is valid for target date.

        SGP4 precision: <1km within ±1-2 days from epoch

        Args:
            target_date: Target propagation date
            max_days: Maximum days from epoch (default: 2)

        Returns:
            True if within precision range
        """
        delta = abs((target_date - self.epoch).days)
        return delta <= max_days

    def __repr__(self):
        return (f"TLE(sat_id={self.satellite_id}, name={self.satellite_name}, "
                f"epoch={self.epoch.strftime('%Y-%m-%d')})")


class TLELoader:
    """
    TLE Data Loader and Manager.

    Supports two loading strategies:
    1. Single TLE file: For short-duration propagation (1-2 days)
    2. Multiple TLE files: For 30-day precision strategy (30 TLE × 1 day)

    SOURCE:
    - 30-day strategy: Maintains SGP4 accuracy <1km
    - Each TLE propagates 1 day from its epoch
    """

    def __init__(self, tle_directory: str = None, file_pattern: str = "*.tle",
                 tle_sources: List[Tuple[str, str]] = None):
        """
        Initialize TLE Loader.

        Supports two modes:
        1. Single directory (backward compatible):
           TLELoader(tle_directory="/path/to/tles", file_pattern="*.tle")

        2. Multiple directories (multi-constellation):
           TLELoader(tle_sources=[
               ("/path/to/starlink", "starlink_*.tle"),
               ("/path/to/oneweb", "oneweb_*.tle")
           ])

        Args:
            tle_directory: Single directory containing TLE files (legacy mode)
            file_pattern: Glob pattern for TLE files (default: "*.tle")
            tle_sources: List of (directory, pattern) tuples for multi-constellation
        """
        # Storage: {satellite_id: [TLE1, TLE2, ...]} sorted by epoch
        self.tle_data: Dict[str, List[TLE]] = {}

        # Name to ID mapping: {satellite_name: satellite_id}
        self.name_to_id: Dict[str, str] = {}

        # Configure TLE sources
        if tle_sources:
            # Multi-constellation mode
            self.tle_sources = [(Path(d), p) for d, p in tle_sources]
            # Validate all directories exist
            for directory, pattern in self.tle_sources:
                if not directory.exists():
                    raise FileNotFoundError(f"TLE directory not found: {directory}")
        elif tle_directory:
            # Single directory mode (backward compatible)
            self.tle_directory = Path(tle_directory)
            if not self.tle_directory.exists():
                raise FileNotFoundError(f"TLE directory not found: {tle_directory}")
            self.tle_sources = [(self.tle_directory, file_pattern)]
        else:
            raise ValueError("Must provide either tle_directory or tle_sources")

    def load_all_tles(self) -> int:
        """
        Load all TLE files from configured directories.

        Returns:
            Number of TLE files loaded
        """
        loaded_count = 0

        # Load from all configured sources
        for directory, pattern in self.tle_sources:
            tle_files = sorted(directory.glob(pattern))

            if not tle_files:
                print(f"⚠️ No TLE files found matching pattern '{pattern}' in {directory}")
                continue

            for tle_file in tle_files:
                self._load_tle_file(tle_file)
                loaded_count += 1

        if loaded_count == 0:
            raise FileNotFoundError(
                f"No TLE files found in any configured source: {self.tle_sources}"
            )

        # Sort TLEs by epoch for each satellite
        for sat_id in self.tle_data:
            self.tle_data[sat_id].sort(key=lambda tle: tle.epoch)

        return loaded_count

    def _load_tle_file(self, file_path: Path):
        """
        Load a single TLE file.

        TLE format (3-line):
            Line 0: Satellite name
            Line 1: TLE line 1 (69 characters)
            Line 2: TLE line 2 (69 characters)

        SOURCE: https://celestrak.org/NORAD/documentation/tle-fmt.php
        """
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # Process in groups of 3 (name, line1, line2)
        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                break

            name = lines[i]
            line1 = lines[i + 1]
            line2 = lines[i + 2]

            # Validate TLE format
            if not (line1.startswith('1 ') and line2.startswith('2 ')):
                continue

            # Extract satellite ID from line 1 (columns 3-7)
            sat_id = line1[2:7].strip()

            # Create TLE object
            tle = TLE(sat_id, name, line1, line2, str(file_path))

            # Store by satellite ID
            if sat_id not in self.tle_data:
                self.tle_data[sat_id] = []
            self.tle_data[sat_id].append(tle)

            # Store name to ID mapping
            if name.strip():
                self.name_to_id[name.strip()] = sat_id

    def _resolve_satellite_id(self, satellite_id: str) -> str:
        """
        Resolve satellite identifier (name or NORAD ID) to NORAD ID.

        Args:
            satellite_id: Either satellite name (e.g., "STARLINK-1008") or NORAD ID (e.g., "44714")

        Returns:
            NORAD catalog number
        """
        # Check if it's already a NORAD ID
        if satellite_id in self.tle_data:
            return satellite_id

        # Try to resolve from name
        if satellite_id in self.name_to_id:
            return self.name_to_id[satellite_id]

        # Not found
        return satellite_id  # Return as-is, will fail later with clear error

    def get_tle_for_date(self, satellite_id: str, target_date: datetime) -> Optional[TLE]:
        """
        Get the best TLE for a target date.

        Strategy: Find TLE with epoch closest to target date (within ±2 days)

        Args:
            satellite_id: Satellite name (e.g., "STARLINK-1008") or NORAD catalog number (e.g., "44714")
            target_date: Target propagation date

        Returns:
            TLE object, or None if not found
        """
        # Resolve name to ID if needed
        satellite_id = self._resolve_satellite_id(satellite_id)

        if satellite_id not in self.tle_data:
            return None

        tles = self.tle_data[satellite_id]

        # Find TLE with closest epoch
        best_tle = None
        min_delta = float('inf')

        for tle in tles:
            delta = abs((target_date - tle.epoch).total_seconds())
            if delta < min_delta and tle.is_valid_for_date(target_date):
                min_delta = delta
                best_tle = tle

        return best_tle

    def get_tle_sequence_for_period(self, satellite_id: str,
                                   start_date: datetime,
                                   end_date: datetime) -> List[Tuple[datetime, datetime, TLE]]:
        """
        Get TLE sequence for a time period (30-day strategy).

        Returns list of (start, end, tle) tuples, where each TLE is used
        for propagation within its valid range (±1-2 days from epoch).

        Example for 30 days:
            [(day0, day1, TLE1), (day1, day2, TLE2), ..., (day29, day30, TLE30)]

        Args:
            satellite_id: Satellite name (e.g., "STARLINK-1008") or NORAD catalog number
            start_date: Start date
            end_date: End date

        Returns:
            List of (segment_start, segment_end, tle) tuples
        """
        # Resolve name to ID if needed
        satellite_id = self._resolve_satellite_id(satellite_id)

        if satellite_id not in self.tle_data:
            return []

        tles = self.tle_data[satellite_id]
        segments = []

        current_date = start_date
        while current_date < end_date:
            # Find TLE for current date
            tle = self.get_tle_for_date(satellite_id, current_date)

            if tle is None:
                # No valid TLE, skip this period
                current_date += timedelta(days=1)
                continue

            # Determine segment end (1 day or until next TLE)
            segment_end = min(current_date + timedelta(days=1), end_date)

            segments.append((current_date, segment_end, tle))
            current_date = segment_end

        return segments

    def get_available_satellites(self) -> List[str]:
        """
        Get list of available satellite IDs.

        Returns:
            List of NORAD catalog numbers
        """
        return sorted(self.tle_data.keys())

    def get_tle_count(self, satellite_id: str = None) -> int:
        """
        Get TLE count.

        Args:
            satellite_id: Specific satellite, or None for total

        Returns:
            Number of TLEs
        """
        if satellite_id:
            return len(self.tle_data.get(satellite_id, []))
        else:
            return sum(len(tles) for tles in self.tle_data.values())

    def get_epoch_range(self, satellite_id: str = None) -> Optional[Tuple[datetime, datetime]]:
        """
        Get epoch date range.

        Args:
            satellite_id: Specific satellite, or None for all

        Returns:
            (earliest_epoch, latest_epoch) or None
        """
        if satellite_id:
            tles = self.tle_data.get(satellite_id, [])
        else:
            tles = [tle for tle_list in self.tle_data.values() for tle in tle_list]

        if not tles:
            return None

        epochs = [tle.epoch for tle in tles]
        return (min(epochs), max(epochs))

    def validate_coverage(self, start_date: datetime, end_date: datetime,
                         satellite_ids: List[str] = None) -> Dict[str, bool]:
        """
        Validate TLE coverage for a time period.

        Args:
            start_date: Start date
            end_date: End date
            satellite_ids: Satellites to check, or None for all

        Returns:
            {satellite_id: has_coverage}
        """
        if satellite_ids is None:
            satellite_ids = self.get_available_satellites()

        coverage = {}
        for sat_id in satellite_ids:
            segments = self.get_tle_sequence_for_period(sat_id, start_date, end_date)
            # Check if we have continuous coverage
            total_coverage = sum((end - start).total_seconds()
                               for start, end, _ in segments)
            expected_coverage = (end_date - start_date).total_seconds()
            coverage[sat_id] = (total_coverage >= expected_coverage * 0.95)  # 95% threshold

        return coverage

    def __repr__(self):
        return (f"TLELoader(directory={self.tle_directory}, "
                f"satellites={len(self.tle_data)}, "
                f"total_tles={self.get_tle_count()})")


# Example usage
if __name__ == "__main__":
    # Example: Load TLEs for 30-day strategy
    loader = TLELoader(
        tle_directory="/home/sat/satellite/orbit-engine/data/tle_data/starlink/tle",
        file_pattern="starlink_*.tle"
    )

    print("Loading TLE files...")
    count = loader.load_all_tles()
    print(f"✅ Loaded {count} TLE files")
    print(f"   Available satellites: {len(loader.get_available_satellites())}")
    print(f"   Total TLEs: {loader.get_tle_count()}")

    # Check epoch range
    epoch_range = loader.get_epoch_range()
    if epoch_range:
        print(f"   Epoch range: {epoch_range[0].date()} to {epoch_range[1].date()}")

    # Example: Get TLE sequence for 30 days
    sat_id = loader.get_available_satellites()[0]
    start = datetime(2025, 10, 1)
    end = datetime(2025, 10, 31)

    segments = loader.get_tle_sequence_for_period(sat_id, start, end)
    print(f"\n📅 TLE sequence for satellite {sat_id} (30 days):")
    print(f"   Segments: {len(segments)}")

    # Validate coverage
    coverage = loader.validate_coverage(start, end, [sat_id])
    print(f"   Coverage: {'✅ Complete' if coverage[sat_id] else '❌ Incomplete'}")
