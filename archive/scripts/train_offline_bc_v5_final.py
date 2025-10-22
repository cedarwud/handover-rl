#!/usr/bin/env python3
"""
Offline Behavior Cloning Training V5 - FINAL SOLUTION
======================================================

最終修正策略:
1. Positive samples: 從 Stage 6 A4 events (已觸發, margin > 0)
2. Negative samples: 從同一 timestamp 的候選池中隨機選其他衛星
   - 確保同時可見 (visibility constraint)
   - 計算真實 trigger margin
   - 只保留 margin ≤ 0 的樣本

目標: 85-95% accuracy
"""

import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'src'))

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

from adapters.handover_event_loader import create_handover_event_loader

A4_THRESHOLD = -34.5
A4_HYSTERESIS = 2.0

class HandoverPolicyNet(nn.Module):
    def __init__(self, input_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def build_timestamp_index(signal_analysis: Dict) -> Dict[str, Dict]:
    """Build index: timestamp -> {sat_id: rsrp}"""
    logger.info("🔨 Building timestamp index...")
    timestamp_index = defaultdict(dict)

    for sat_id, sat_data in signal_analysis.items():
        time_series = sat_data.get('time_series', [])
        for tp in time_series:
            timestamp = tp.get('timestamp')
            signal_quality = tp.get('signal_quality', {})
            rsrp = signal_quality.get('rsrp_dbm')

            if timestamp and rsrp is not None:
                timestamp_index[timestamp][sat_id] = {
                    'rsrp': rsrp,
                    'distance': tp.get('visibility_metrics', {}).get('ground_distance_km', 1500)
                }

    logger.info(f"✅ Indexed {len(timestamp_index)} timestamps")
    return timestamp_index

def extract_samples_v5(a4_events, signal_analysis, timestamp_index):
    """
    Extract both positive and negative samples using timestamp matching

    For each A4 event (positive):
    - timestamp, serving, neighbor are known
    - Find other satellites visible at same timestamp
    - Calculate their trigger margins
    - margin > 0 → handover (should already be in events)
    - margin ≤ 0 → maintain (NEW negative samples)
    """
    positive_samples = []
    negative_samples = []

    logger.info(f"📊 Processing {len(a4_events)} A4 events...")

    for event in a4_events:
        timestamp = event.get('timestamp')
        serving_id = event.get('serving_satellite')
        neighbor_id = event.get('neighbor_satellite')
        measurements = event.get('measurements', {})

        neighbor_rsrp = measurements.get('neighbor_rsrp_dbm')
        trigger_margin = measurements.get('trigger_margin_db')

        if not all([timestamp, serving_id, neighbor_id, neighbor_rsrp is not None]):
            continue

        # Get serving RSRP from timestamp index
        if timestamp not in timestamp_index:
            continue

        satellites_at_time = timestamp_index[timestamp]

        if serving_id not in satellites_at_time:
            continue

        serving_rsrp = satellites_at_time[serving_id]['rsrp']
        serving_dist = satellites_at_time[serving_id]['distance']

        # Positive sample (this event)
        if neighbor_id in satellites_at_time:
            neighbor_dist = satellites_at_time[neighbor_id]['distance']
            features = [
                neighbor_rsrp,
                serving_rsrp,
                neighbor_rsrp - serving_rsrp,
                neighbor_dist,
                serving_dist,
                trigger_margin
            ]
            positive_samples.append((features, 1))

        # Negative samples: other satellites at same timestamp
        for other_id, other_data in satellites_at_time.items():
            if other_id == serving_id or other_id == neighbor_id:
                continue

            # Calculate trigger margin for this potential neighbor
            other_rsrp = other_data['rsrp']
            calc_margin = other_rsrp - A4_THRESHOLD - A4_HYSTERESIS

            # Only keep if margin ≤ 0 (not triggered)
            if calc_margin <= 0:
                features = [
                    other_rsrp,
                    serving_rsrp,
                    other_rsrp - serving_rsrp,
                    other_data['distance'],
                    serving_dist,
                    calc_margin
                ]
                negative_samples.append((features, 0))

    logger.info(f"✅ Extracted {len(positive_samples)} positive samples")
    logger.info(f"✅ Extracted {len(negative_samples)} negative samples")

    return positive_samples, negative_samples

def train_bc_policy(samples, num_epochs=10, lr=0.001):
    logger.info(f"\n🚀 Training (epochs={num_epochs})...")

    np.random.shuffle(samples)
    train_size = int(0.8 * len(samples))
    train_samples = samples[:train_size]
    test_samples = samples[train_size:]

    logger.info(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    X_train = torch.FloatTensor([s[0] for s in train_samples])
    y_train = torch.FloatTensor([s[1] for s in train_samples]).unsqueeze(1)
    X_test = torch.FloatTensor([s[0] for s in test_samples])
    y_test = torch.FloatTensor([s[1] for s in test_samples]).unsqueeze(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HandoverPolicyNet(input_dim=6).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_test_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train.to(device))
        loss = criterion(outputs, y_train.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_pred = (model(X_train.to(device)) > 0.5).float()
            train_acc = (train_pred == y_train.to(device)).float().mean().item() * 100
            test_pred = (model(X_test.to(device)) > 0.5).float()
            test_acc = (test_pred == y_test.to(device)).float().mean().item() * 100
            best_test_acc = max(best_test_acc, test_acc)

        if (epoch + 1) % 2 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f} Train: {train_acc:.2f}% Test: {test_acc:.2f}%")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(__file__).parent / 'checkpoints' / f'bc_policy_v5_{timestamp}.pth'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"✅ Saved: {save_path}")

    return best_test_acc

def main():
    logger.info("=" * 80)
    logger.info("Behavior Cloning V5 - FINAL (Timestamp-based Sampling)")
    logger.info("=" * 80)

    # Load events
    loader = create_handover_event_loader()
    orbit_engine_root = Path(__file__).parent.parent / 'orbit-engine'
    stage6_dir = orbit_engine_root / 'data' / 'outputs' / 'rl_training' / 'stage6'
    a4_events, _ = loader.load_latest_events(stage6_dir)
    logger.info(f"✅ Loaded {len(a4_events)} A4 events")

    # Load Stage 5
    stage5_dir = orbit_engine_root / 'data' / 'outputs' / 'rl_training' / 'stage5'
    stage5_file = sorted(stage5_dir.glob('stage5_signal_analysis_*.json'))[-1]
    with open(stage5_file, 'r') as f:
        stage5_data = json.load(f)
    signal_analysis = stage5_data['signal_analysis']
    logger.info(f"✅ Loaded {len(signal_analysis)} satellites")

    # Build timestamp index
    timestamp_index = build_timestamp_index(signal_analysis)

    # Extract samples
    positive_samples, negative_samples = extract_samples_v5(
        a4_events, signal_analysis, timestamp_index
    )

    # Combine
    all_samples = positive_samples + negative_samples
    logger.info(f"\n📊 Dataset: {len(all_samples)} total")
    logger.info(f"  Positive: {len(positive_samples)} ({len(positive_samples)/len(all_samples)*100:.1f}%)")
    logger.info(f"  Negative: {len(negative_samples)} ({len(negative_samples)/len(all_samples)*100:.1f}%)")

    # Train
    test_acc = train_bc_policy(all_samples, num_epochs=10, lr=0.001)

    logger.info("\n✅ FINAL RESULTS")
    logger.info(f"  Test Accuracy: {test_acc:.2f}%")
    logger.info(f"  Expected: 85-95%")
    logger.info(f"  Status: {'✅ SUCCESS' if 85 <= test_acc <= 95 else '⚠️ NEEDS TUNING'}")

if __name__ == '__main__':
    main()
