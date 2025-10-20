#!/usr/bin/env python3
"""
Handover Event Loader - Load A4/D2 Events from Orbit-Engine Stage 6 Output

✅ REAL DATA ONLY - Loads official 3GPP handover events from orbit-engine
✅ Academic Grade A - Full compliance with 3GPP TS 38.331 v18.5.1

Purpose:
- Load A4 (threshold-based) and D2 (distance-based) handover events
- Provide baseline handover strategies for RL training
- Extract state-action pairs from real satellite handover scenarios

Data Source:
- orbit-engine Stage 6 output: stage6_research_optimization_*.json
- Generated using official 3GPP standards and real TLE data
- Contains complete event metadata and measurements

Design Pattern: Data Loader
- Simple interface: load_events(file_path) → A4/D2 events
- No data transformation - preserves original event structure
- Validation: ensures all required fields exist

Created: 2025-10-20
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class HandoverEventLoader:
    """
    Load A4 and D2 handover events from orbit-engine Stage 6 output.

    Usage:
        loader = HandoverEventLoader()
        a4_events, d2_events = loader.load_events(stage6_file)
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize handover event loader.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def load_events(
        self,
        stage6_file: Path
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Load A4 and D2 events from Stage 6 output.

        Args:
            stage6_file: Path to stage6_research_optimization_*.json

        Returns:
            Tuple of (a4_events, d2_events)

        Raises:
            FileNotFoundError: If stage6_file doesn't exist
            ValueError: If data structure is invalid
        """
        if not stage6_file.exists():
            raise FileNotFoundError(
                f"Stage 6 output not found: {stage6_file}\n"
                f"Please run orbit-engine Stage 6 first to generate handover events"
            )

        self.logger.info(f"📥 Loading handover events from: {stage6_file}")

        # Load JSON
        with open(stage6_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate structure
        if 'gpp_events' not in data:
            raise ValueError(
                f"Missing 'gpp_events' field in Stage 6 output\n"
                f"File: {stage6_file}"
            )

        gpp_events = data['gpp_events']

        # Extract A4 and D2 events
        a4_events = gpp_events.get('a4_events', [])
        d2_events = gpp_events.get('d2_events', [])

        self.logger.info(f"✅ Loaded {len(a4_events)} A4 events, {len(d2_events)} D2 events")

        # Validate events
        self._validate_a4_events(a4_events)
        self._validate_d2_events(d2_events)

        return a4_events, d2_events

    def _validate_a4_events(self, events: List[Dict[str, Any]]) -> None:
        """
        Validate A4 event structure.

        Required fields:
        - event_type: "A4"
        - timestamp: ISO 8601 timestamp
        - serving_satellite: satellite ID
        - measurements: RSRP measurements
        - gpp_parameters: 3GPP parameters

        Raises:
            ValueError: If any event is invalid
        """
        if not events:
            return  # Empty list is valid

        for i, event in enumerate(events):
            # Check required fields
            required_fields = [
                'event_type', 'timestamp', 'serving_satellite',
                'measurements', 'gpp_parameters'
            ]

            for field in required_fields:
                if field not in event:
                    raise ValueError(
                        f"A4 event {i} missing required field: {field}\n"
                        f"Event: {event}"
                    )

            # Check event_type
            if event['event_type'] != 'A4':
                raise ValueError(
                    f"A4 event {i} has wrong event_type: {event['event_type']}"
                )

            # Check measurements
            measurements = event['measurements']
            required_measurements = ['neighbor_rsrp_dbm', 'threshold_dbm']

            for field in required_measurements:
                if field not in measurements:
                    raise ValueError(
                        f"A4 event {i} missing {field} in measurements"
                    )

    def _validate_d2_events(self, events: List[Dict[str, Any]]) -> None:
        """
        Validate D2 event structure.

        Required fields:
        - event_type: "D2"
        - timestamp: ISO 8601 timestamp
        - serving_satellite: satellite ID
        - neighbor_satellite: satellite ID
        - measurements: Distance measurements
        - gpp_parameters: 3GPP parameters

        Raises:
            ValueError: If any event is invalid
        """
        if not events:
            return  # Empty list is valid

        for i, event in enumerate(events):
            # Check required fields
            required_fields = [
                'event_type', 'timestamp', 'serving_satellite',
                'neighbor_satellite', 'measurements', 'gpp_parameters'
            ]

            for field in required_fields:
                if field not in event:
                    raise ValueError(
                        f"D2 event {i} missing required field: {field}\n"
                        f"Event: {event}"
                    )

            # Check event_type
            if event['event_type'] != 'D2':
                raise ValueError(
                    f"D2 event {i} has wrong event_type: {event['event_type']}"
                )

            # Check measurements
            measurements = event['measurements']
            required_measurements = [
                'serving_ground_distance_km',
                'neighbor_ground_distance_km'
            ]

            for field in required_measurements:
                if field not in measurements:
                    raise ValueError(
                        f"D2 event {i} missing {field} in measurements"
                    )

    def load_latest_events(
        self,
        stage6_dir: Path
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Load events from the latest Stage 6 output file.

        Args:
            stage6_dir: Directory containing Stage 6 outputs

        Returns:
            Tuple of (a4_events, d2_events)

        Raises:
            FileNotFoundError: If no Stage 6 output found
        """
        # Find latest file
        pattern = "stage6_research_optimization_*.json"
        files = list(stage6_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(
                f"No Stage 6 output found in: {stage6_dir}\n"
                f"Looking for pattern: {pattern}\n"
                f"Please run orbit-engine Stage 6 first"
            )

        # Sort by modification time
        latest_file = max(files, key=lambda f: f.stat().st_mtime)

        self.logger.info(f"📂 Using latest Stage 6 output: {latest_file.name}")

        return self.load_events(latest_file)

    def extract_baseline_policy(
        self,
        a4_events: List[Dict[str, Any]],
        d2_events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract baseline handover policy from A4/D2 events.

        Baseline Policy:
        - A4: Handover when serving RSRP < threshold (3GPP standard)
        - D2: Handover when neighbor is significantly closer (distance-based)

        Args:
            a4_events: List of A4 events
            d2_events: List of D2 events

        Returns:
            Baseline policy dictionary with thresholds and statistics
        """
        policy = {
            'a4_policy': self._extract_a4_policy(a4_events),
            'd2_policy': self._extract_d2_policy(d2_events),
            'total_events': len(a4_events) + len(d2_events),
            'extraction_timestamp': datetime.utcnow().isoformat()
        }

        return policy

    def _extract_a4_policy(
        self,
        events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract A4 baseline policy from events.

        A4 Event: "Neighbour becomes better than threshold"
        - Uses neighbor_rsrp_dbm and threshold_dbm from measurements
        """
        if not events:
            return {
                'enabled': False,
                'reason': 'no_a4_events'
            }

        # Extract neighbor RSRP and thresholds from events
        neighbor_rsrps = []
        thresholds = []

        for event in events:
            measurements = event['measurements']
            neighbor_rsrps.append(measurements['neighbor_rsrp_dbm'])
            thresholds.append(measurements['threshold_dbm'])

        # Calculate statistics
        import numpy as np
        neighbor_array = np.array(neighbor_rsrps)
        threshold_array = np.array(thresholds)

        return {
            'enabled': True,
            'event_count': len(events),
            'neighbor_rsrp_statistics': {
                'mean': float(np.mean(neighbor_array)),
                'median': float(np.median(neighbor_array)),
                'min': float(np.min(neighbor_array)),
                'max': float(np.max(neighbor_array)),
                'std': float(np.std(neighbor_array))
            },
            'threshold_statistics': {
                'mean': float(np.mean(threshold_array)),
                'median': float(np.median(threshold_array)),
                'min': float(np.min(threshold_array)),
                'max': float(np.max(threshold_array))
            },
            'recommended_threshold_dbm': float(np.median(threshold_array)),
            'standard_reference': '3GPP_TS_38.331_v18.5.1_Section_5.5.4.5'
        }

    def _extract_d2_policy(
        self,
        events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract D2 baseline policy from events."""
        if not events:
            return {
                'enabled': False,
                'reason': 'no_d2_events'
            }

        # Extract distance thresholds from events
        serving_distances = []
        neighbor_distances = []

        for event in events:
            measurements = event['measurements']
            serving_distances.append(measurements['serving_ground_distance_km'])
            neighbor_distances.append(measurements['neighbor_ground_distance_km'])

        # Calculate statistics
        import numpy as np
        serving_array = np.array(serving_distances)
        neighbor_array = np.array(neighbor_distances)

        return {
            'enabled': True,
            'event_count': len(events),
            'distance_statistics': {
                'serving_mean_km': float(np.mean(serving_array)),
                'neighbor_mean_km': float(np.mean(neighbor_array)),
                'improvement_mean_km': float(np.mean(serving_array - neighbor_array))
            },
            'standard_reference': '3GPP_TS_38.331_v18.5.1_Section_5.5.4.15a'
        }


# Factory function
def create_handover_event_loader(
    logger: Optional[logging.Logger] = None
) -> HandoverEventLoader:
    """
    Create handover event loader instance.

    Args:
        logger: Optional logger instance

    Returns:
        HandoverEventLoader instance
    """
    return HandoverEventLoader(logger=logger)
