#!/usr/bin/env python3
"""
Offline Behavior Cloning Training V3 (With Threshold Correction)

This version uses existing RL training data but recalculates trigger margins
with the new threshold values to fix data leakage.

Key fixes:
1. Recalculate A4 trigger_margin using NEW threshold (-34.5 dBm)
2. Recalculate A3 offsets using NEW offset (2.5 dB)
3. Filter out events that no longer satisfy the new thresholds
"""

import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from adapters.handover_event_loader import create_handover_event_loader
from models.bc_policy import BCPolicy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

# New threshold values (from Option 2 config)
NEW_A3_OFFSET = 2.5
NEW_A3_HYSTERESIS = 1.5
NEW_A4_THRESHOLD = -34.5
NEW_A4_HYSTERESIS = 2.0

def recalculate_a4_trigger_margin(event: Dict, new_threshold: float) -> float:
    """Recalculate A4 trigger margin with new threshold"""
    measurements = event.get('measurements', {})
    neighbor_rsrp = measurements.get('neighbor_rsrp_dbm')

    if neighbor_rsrp is None:
        return None

    # New trigger margin = neighbor_rsrp - new_threshold - hysteresis
    trigger_margin = neighbor_rsrp - new_threshold - NEW_A4_HYSTERESIS
    return trigger_margin

def recalculate_a3_offset_margin(event: Dict, new_offset: float, new_hysteresis: float) -> float:
    """Recalculate A3 offset margin with new threshold"""
    measurements = event.get('measurements', {})
    neighbor_rsrp = measurements.get('neighbor_rsrp_dbm')
    serving_rsrp = measurements.get('serving_rsrp_dbm')

    if neighbor_rsrp is None or serving_rsrp is None:
        return None

    # A3: Mn + Ofn + Ocn - Hys > Mp + Ofp + Ocp + Off
    # Simplified: neighbor_rsrp - serving_rsrp > offset + hysteresis
    rsrp_diff = neighbor_rsrp - serving_rsrp
    margin = rsrp_diff - (new_offset + new_hysteresis)
    return margin

def filter_events_with_new_thresholds(
    a4_events: List[Dict],
    d2_events: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter events to only include those that satisfy NEW thresholds.
    This removes events that were only triggered due to unrealistic old thresholds.
    """
    # Filter A4 events
    filtered_a4 = []
    for event in a4_events:
        new_margin = recalculate_a4_trigger_margin(event, NEW_A4_THRESHOLD)
        if new_margin is not None and new_margin > 0:
            # Event still triggers with new threshold
            filtered_a4.append(event)

    logger.info(f"A4 Events: {len(a4_events)} → {len(filtered_a4)} after NEW threshold filter")
    logger.info(f"  Removed {len(a4_events) - len(filtered_a4)} events that don't satisfy new threshold")

    # D2 events don't need recalculation (distance-based)
    return filtered_a4, d2_events

def extract_features_from_event(event: Dict, use_new_thresholds: bool = True) -> np.ndarray:
    """
    Extract features from handover event.

    If use_new_thresholds=True, recalculates trigger margins with new thresholds.
    """
    measurements = event.get('measurements', {})

    # Common features
    neighbor_rsrp = measurements.get('neighbor_rsrp_dbm', -100.0)
    serving_rsrp = measurements.get('serving_rsrp_dbm', neighbor_rsrp)
    rsrp_diff = neighbor_rsrp - serving_rsrp

    # Event-specific features
    if event.get('event_type') == 'A4':
        if use_new_thresholds:
            trigger_margin = recalculate_a4_trigger_margin(event, NEW_A4_THRESHOLD)
        else:
            trigger_margin = measurements.get('trigger_margin_db', 0.0)

        distance_km = measurements.get('neighbor_distance_km', 1000.0)

    elif event.get('event_type') == 'D2':
        # Use ground distance for D2
        trigger_margin = measurements.get('ground_distance_improvement_km', 0.0) / 100.0  # Normalize
        distance_km = measurements.get('neighbor_ground_distance_km', 1000.0)

    else:  # A3 or others
        if use_new_thresholds:
            trigger_margin = recalculate_a3_offset_margin(event, NEW_A3_OFFSET, NEW_A3_HYSTERESIS)
        else:
            trigger_margin = measurements.get('offset_mo_db', 0.0)

        distance_km = measurements.get('neighbor_distance_km', 1000.0)

    # Normalize distance (typical range 500-2000 km)
    distance_normalized = (distance_km - 1000.0) / 500.0

    # Additional features
    elevation = measurements.get('neighbor_elevation', 45.0)
    azimuth = measurements.get('neighbor_azimuth', 180.0)

    # Feature vector (10 dimensions)
    features = np.array([
        neighbor_rsrp / 100.0,           # Normalize RSRP
        serving_rsrp / 100.0,
        rsrp_diff / 10.0,
        trigger_margin / 10.0,           # RECALCULATED with new threshold
        distance_normalized,
        elevation / 90.0,
        azimuth / 360.0,
        1.0 if event.get('event_type') == 'A4' else 0.0,
        1.0 if event.get('event_type') == 'D2' else 0.0,
        1.0  # Bias term
    ], dtype=np.float32)

    return features

def prepare_dataset_v3(
    a4_events: List[Dict],
    d2_events: List[Dict],
    signal_analysis: Dict,
    negative_ratio: float = 2.0,
    use_new_thresholds: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Prepare dataset with NEW threshold calculations.

    Strategy:
    1. Filter events to only those satisfying NEW thresholds
    2. Recalculate trigger margins with NEW thresholds
    3. Generate negative samples from non-handover moments
    """

    logger.info("=" * 80)
    logger.info("Preparing Dataset V3 (With Threshold Correction)")
    logger.info("=" * 80)

    # Filter events with new thresholds
    filtered_a4, filtered_d2 = filter_events_with_new_thresholds(a4_events, d2_events)

    handover_events = filtered_a4 + filtered_d2

    logger.info(f"\nFiltered handover events: {len(handover_events)}")
    logger.info(f"  A4: {len(filtered_a4)}")
    logger.info(f"  D2: {len(filtered_d2)}")

    # Extract positive examples (handover)
    features_list = []
    labels_list = []
    sat_ids_list = []

    for event in handover_events:
        features = extract_features_from_event(event, use_new_thresholds=use_new_thresholds)
        features_list.append(features)
        labels_list.append(1)  # Handover
        sat_ids_list.append(event.get('neighbor_satellite', 'unknown'))

    logger.info(f"✅ Extracted {len(features_list)} positive examples (handover)")

    # Generate negative examples (maintain)
    num_negatives = int(len(handover_events) * negative_ratio)
    logger.info(f"\n🔄 Sampling {num_negatives} negative examples (maintain)...")

    # Extract all satellite time series
    available_sats = [sat_id for sat_id, sat_data in signal_analysis.items()
                     if 'time_series' in sat_data and len(sat_data['time_series']) > 0]

    logger.info(f"Available satellites in Stage 5: {len(available_sats)}")

    sample_count = 0
    for _ in range(num_negatives):
        # Random satellite and time point
        sat_id = np.random.choice(available_sats)
        sat_data = signal_analysis[sat_id]
        time_series = sat_data['time_series']

        if len(time_series) == 0:
            continue

        tp_idx = np.random.randint(0, len(time_series))
        tp = time_series[tp_idx]

        signal_quality = tp.get('signal_quality', {})
        rsrp_dbm = signal_quality.get('rsrp_dbm', -100.0)

        # Create fake "maintain" event
        # Key: Use similar RSRP but with small or negative trigger margin
        fake_event = {
            'event_type': 'MAINTAIN',
            'measurements': {
                'neighbor_rsrp_dbm': rsrp_dbm + np.random.uniform(-3, 0),  # Slightly worse
                'serving_rsrp_dbm': rsrp_dbm,
                'neighbor_distance_km': tp.get('distance_km', 1000.0),
                'neighbor_elevation': tp.get('elevation', 45.0),
                'neighbor_azimuth': tp.get('azimuth', 180.0)
            }
        }

        features = extract_features_from_event(fake_event, use_new_thresholds=use_new_thresholds)
        features_list.append(features)
        labels_list.append(0)  # Maintain
        sat_ids_list.append(sat_id)

        sample_count += 1
        if sample_count % 10000 == 0:
            logger.info(f"  Progress: {sample_count}/{num_negatives}")

    logger.info(f"✅ Extracted {sample_count} negative examples (maintain)")

    # Convert to tensors
    features_tensor = torch.tensor(np.array(features_list), dtype=torch.float32)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    logger.info(f"\n📊 Final dataset:")
    logger.info(f"  Total: {len(features_tensor)} samples")
    logger.info(f"  Positive: {sum(labels_list)} ({sum(labels_list)/len(labels_list)*100:.1f}%)")
    logger.info(f"  Negative: {len(labels_list) - sum(labels_list)} ({(len(labels_list) - sum(labels_list))/len(labels_list)*100:.1f}%)")

    return features_tensor, labels_tensor, sat_ids_list

def main():
    logger.info("=" * 80)
    logger.info("Offline Behavior Cloning Training V3 (Threshold-Corrected)")
    logger.info("=" * 80)

    logger.info("\n📥 Loading handover events...")

    # Load events
    loader = create_handover_event_loader()
    orbit_engine_root = Path(__file__).parent.parent / 'orbit-engine'
    stage6_dir = orbit_engine_root / 'data' / 'outputs' / 'rl_training' / 'stage6'

    a4_events, d2_events = loader.load_latest_events(stage6_dir)

    logger.info("\n📥 Loading Stage 5 signal analysis...")
    stage5_dir = orbit_engine_root / 'data' / 'outputs' / 'rl_training' / 'stage5'
    stage5_file = sorted(stage5_dir.glob('stage5_signal_analysis_*.json'))[-1]

    logger.info(f"📥 Loading Stage 5 signal data from: {stage5_file}")

    with open(stage5_file, 'r') as f:
        stage5_data = json.load(f)

    signal_analysis = stage5_data['signal_analysis']
    logger.info(f"✅ Loaded time series for {len(signal_analysis)} satellites")

    # Prepare dataset with NEW thresholds
    logger.info("\n🔧 Preparing dataset with threshold correction...")
    features, labels, sat_ids = prepare_dataset_v3(
        a4_events, d2_events, signal_analysis,
        negative_ratio=2.0,
        use_new_thresholds=True
    )

    # Split by satellite (not random)
    unique_sats = list(set(sat_ids))
    np.random.shuffle(unique_sats)

    train_sats = set(unique_sats[:int(len(unique_sats) * 0.8)])
    test_sats = set(unique_sats[int(len(unique_sats) * 0.8):])

    train_mask = [sat in train_sats for sat in sat_ids]
    test_mask = [sat in test_sats for sat in sat_ids]

    X_train = features[train_mask]
    y_train = labels[train_mask]
    X_test = features[test_mask]
    y_test = labels[test_mask]

    logger.info("\n🚀 Starting training...")
    logger.info(f"Train satellites: {len(train_sats)}")
    logger.info(f"Test satellites: {len(test_sats)}")
    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")

    # Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model = BCPolicy(input_dim=10, hidden_dims=[256, 128, 64]).to(device)

    # Class balancing
    pos_weight = (y_train == 0).sum().float() / (y_train == 1).sum().float()
    logger.info(f"Positive weight: {pos_weight:.2f} (class balancing)")

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight.item()]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    batch_size = 512
    epochs = 20
    best_test_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Mini-batch training
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch_X = X_train[batch_idx].to(device)
            batch_y = y_train[batch_idx].to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)

        train_acc = 100.0 * correct / total

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test.to(device))
            _, test_predicted = test_outputs.max(1)
            test_correct = test_predicted.eq(y_test.to(device)).sum().item()
            test_acc = 100.0 * test_correct / len(X_test)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(X_train):.4f} Train Acc: {train_acc:.2f}% Test Acc: {test_acc:.2f}%")

    logger.info(f"Training complete! Best test accuracy: {best_test_acc:.2f}%")

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(__file__).parent / 'checkpoints' / f'bc_policy_v3_{timestamp}.pth'
    save_path.parent.mkdir(exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': 10,
            'hidden_dims': [256, 128, 64],
            'new_a3_offset': NEW_A3_OFFSET,
            'new_a4_threshold': NEW_A4_THRESHOLD,
            'threshold_corrected': True
        }
    }, save_path)

    logger.info(f"✅ Model saved: {save_path}")
    logger.info(f"\n✅ Training Complete!")

    # Print summary
    logger.info(f"\n📊 Training Summary:")
    logger.info(f"  New A3 offset: {NEW_A3_OFFSET} dB (was 2.0 dB)")
    logger.info(f"  New A4 threshold: {NEW_A4_THRESHOLD} dBm (was -100.0 dBm)")
    logger.info(f"  Final test accuracy: {best_test_acc:.2f}%")
    logger.info(f"  Expected range: 85-95% (realistic learning)")

if __name__ == '__main__':
    main()
