#!/usr/bin/env python3
"""
Offline Behavior Cloning Training - Learn from A4/D2 Handover Events

Train a neural network policy by imitating expert handover decisions
from orbit-engine Stage 6 A4/D2 events.

Method: Supervised Learning (Behavior Cloning)
- Extract (state, action) pairs from handover events
- Train neural network: state → action
- Evaluate: accuracy, precision, recall

Usage:
    python train_offline_bc.py --epochs 50 --batch-size 256

Input Data:
    orbit-engine Stage 6 output (gpp_events_candidate)
    - A4 events: threshold-based handovers
    - D2 events: distance-based handovers

Output:
    - Trained model: checkpoints/bc_policy_*.pth
    - Training metrics: logs/bc_training.log
    - Evaluation report: results/bc_evaluation.json
"""

import sys
import json
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from adapters.handover_event_loader import create_handover_event_loader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bc_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ========== Neural Network Policy ==========

class BCPolicy(nn.Module):
    """
    Behavior Cloning Policy Network

    Architecture: MLP (Multi-Layer Perceptron)
    Input: Satellite state features (RSRP, distance, elevation, etc.)
    Output: Handover decision (binary: handover or maintain)
    """

    def __init__(self, input_dim: int = 10, hidden_dims: List[int] = [256, 128, 64]):
        """
        Initialize BC policy network.

        Args:
            input_dim: Number of input features
            hidden_dims: Hidden layer dimensions
        """
        super(BCPolicy, self).__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)  # Regularization
            ])
            prev_dim = hidden_dim

        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 2))  # 2 classes: maintain, handover

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        return self.network(x)


# ========== Feature Extraction ==========

def extract_features_from_event(event: Dict[str, Any]) -> np.ndarray:
    """
    Extract state features from handover event.

    Features (10 dimensions):
    1. Neighbor RSRP (dBm)
    2. Serving RSRP (if available, else use threshold)
    3. RSRP差值 (neighbor - serving)
    4. Neighbor distance (km)
    5. Serving distance (km)
    6. Distance improvement (serving - neighbor)
    7. Neighbor elevation (deg) - if available
    8. Event type indicator (0=A4, 1=D2)
    9. Trigger margin (dB or km normalized)
    10. Normalized timestamp (0-1 within episode)

    Args:
        event: A4 or D2 event dictionary

    Returns:
        Feature vector (10D numpy array)
    """
    measurements = event['measurements']

    # RSRP features
    neighbor_rsrp = measurements.get('neighbor_rsrp_dbm', -100.0)
    serving_rsrp = measurements.get('serving_rsrp_dbm', measurements.get('threshold_dbm', -100.0))
    rsrp_diff = neighbor_rsrp - serving_rsrp

    # Distance features
    neighbor_dist = measurements.get('neighbor_ground_distance_km', 1000.0)
    serving_dist = measurements.get('serving_ground_distance_km', 1000.0)
    dist_improvement = serving_dist - neighbor_dist

    # Elevation (if available)
    neighbor_elev = measurements.get('neighbor_elevation_deg', 45.0)  # Default to mid-range

    # Event type
    event_type = 0.0 if event['event_type'] == 'A4' else 1.0

    # Trigger margin
    trigger_margin = measurements.get('trigger_margin_db', measurements.get('trigger_margin_km', 0.0))

    # Normalized timestamp (use epoch seconds, normalize to 0-1)
    # For now, use 0.5 as placeholder (middle of episode)
    norm_timestamp = 0.5

    features = np.array([
        neighbor_rsrp / 100.0,      # Normalize RSRP to ~[-1, 0]
        serving_rsrp / 100.0,
        rsrp_diff / 50.0,            # Normalize diff to ~[-2, 2]
        neighbor_dist / 2000.0,      # Normalize distance to [0, 1]
        serving_dist / 2000.0,
        dist_improvement / 2000.0,   # Normalize improvement
        neighbor_elev / 90.0,        # Normalize elevation to [0, 1]
        event_type,
        trigger_margin / 100.0,      # Normalize margin
        norm_timestamp
    ], dtype=np.float32)

    return features


def prepare_dataset(
    a4_events: List[Dict],
    d2_events: List[Dict]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare training dataset from handover events.

    All events are positive examples (handover decision = 1).
    We'll also generate negative examples (maintain decision = 0)
    by perturbing features.

    Args:
        a4_events: List of A4 events
        d2_events: List of D2 events

    Returns:
        (features, labels) tensors
    """
    features_list = []
    labels_list = []

    # Positive examples: actual handover events
    all_events = a4_events + d2_events
    logger.info(f"Processing {len(all_events)} handover events...")

    for event in all_events:
        features = extract_features_from_event(event)
        features_list.append(features)
        labels_list.append(1)  # Handover decision

    # Negative examples: generate by perturbing features
    # Rule: If RSRP_diff < threshold, it's a "maintain" decision
    logger.info(f"Generating negative examples (maintain decisions)...")

    num_negatives = len(all_events)  # Equal number of positives and negatives
    for i in range(num_negatives):
        # Take a random positive example
        idx = np.random.randint(0, len(all_events))
        event = all_events[idx]
        features = extract_features_from_event(event).copy()

        # Perturb RSRP to make neighbor worse than serving
        # Reduce neighbor RSRP or increase serving RSRP
        features[0] -= np.random.uniform(0.1, 0.3)  # Reduce neighbor RSRP
        features[2] = features[0] - features[1]      # Update diff

        features_list.append(features)
        labels_list.append(0)  # Maintain decision

    # Convert to tensors
    features_tensor = torch.tensor(np.array(features_list), dtype=torch.float32)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    logger.info(f"Dataset prepared: {len(features_list)} samples")
    logger.info(f"  Positive (handover): {sum(labels_list)}")
    logger.info(f"  Negative (maintain): {len(labels_list) - sum(labels_list)}")

    return features_tensor, labels_tensor


# ========== Training Function ==========

def train_bc_policy(
    features: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    val_split: float = 0.2
) -> BCPolicy:
    """
    Train behavior cloning policy.

    Args:
        features: Input features tensor
        labels: Target labels tensor
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        val_split: Validation split ratio

    Returns:
        Trained BCPolicy model
    """
    # Split train/val
    num_samples = features.shape[0]
    num_val = int(num_samples * val_split)
    indices = np.random.permutation(num_samples)

    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    train_features = features[train_indices]
    train_labels = labels[train_indices]
    val_features = features[val_indices]
    val_labels = labels[val_indices]

    logger.info(f"Train samples: {len(train_indices)}")
    logger.info(f"Val samples: {len(val_indices)}")

    # Initialize model
    input_dim = features.shape[1]
    model = BCPolicy(input_dim=input_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    logger.info(f"Using device: {device}")
    logger.info(f"Model: {model}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()

        # Shuffle training data
        perm = torch.randperm(train_features.shape[0])
        train_features = train_features[perm]
        train_labels = train_labels[perm]

        epoch_loss = 0.0
        correct = 0
        total = 0

        # Mini-batch training
        for i in range(0, len(train_features), batch_size):
            batch_features = train_features[i:i+batch_size].to(device)
            batch_labels = train_labels[i:i+batch_size].to(device)

            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        train_acc = 100 * correct / total
        avg_loss = epoch_loss / (len(train_features) / batch_size)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_features.to(device))
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_correct = (val_predicted == val_labels.to(device)).sum().item()
            val_acc = 100 * val_correct / len(val_labels)

        # Log progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Loss: {avg_loss:.4f} "
                f"Train Acc: {train_acc:.2f}% "
                f"Val Acc: {val_acc:.2f}%"
            )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_dir = Path('checkpoints')
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / 'bc_policy_best.pth')

    logger.info(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")

    return model


# ========== Main Function ==========

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Offline Behavior Cloning Training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Offline Behavior Cloning Training")
    logger.info("=" * 80)

    # Create output directories
    Path('logs').mkdir(exist_ok=True)
    Path('checkpoints').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)

    # Load handover events
    logger.info("\n📥 Loading handover events from orbit-engine Stage 6...")
    loader = create_handover_event_loader()

    orbit_engine_root = Path(__file__).parent.parent / 'orbit-engine'
    stage6_dir = orbit_engine_root / 'data' / 'outputs' / 'rl_training' / 'stage6'

    a4_events, d2_events = loader.load_latest_events(stage6_dir)
    logger.info(f"✅ Loaded {len(a4_events)} A4 events, {len(d2_events)} D2 events")

    # Prepare dataset
    logger.info("\n🔧 Preparing training dataset...")
    features, labels = prepare_dataset(a4_events, d2_events)

    # Train model
    logger.info("\n🚀 Starting training...")
    logger.info(f"Hyperparameters:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Validation split: {args.val_split}")

    model = train_bc_policy(
        features,
        labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split
    )

    # Save final model
    logger.info("\n💾 Saving final model...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = Path('checkpoints') / f'bc_policy_{timestamp}.pth'
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"✅ Model saved: {final_model_path}")

    # Evaluation
    logger.info("\n📊 Evaluating model...")
    model.eval()
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        outputs = model(features.to(device))
        _, predicted = torch.max(outputs.data, 1)

        correct = (predicted == labels.to(device)).sum().item()
        accuracy = 100 * correct / len(labels)

        # Per-class accuracy
        handover_mask = labels == 1
        maintain_mask = labels == 0

        handover_correct = (predicted[handover_mask] == labels[handover_mask].to(device)).sum().item()
        maintain_correct = (predicted[maintain_mask] == labels[maintain_mask].to(device)).sum().item()

        handover_acc = 100 * handover_correct / handover_mask.sum().item()
        maintain_acc = 100 * maintain_correct / maintain_mask.sum().item()

    # Save evaluation results
    results = {
        'timestamp': timestamp,
        'model_path': str(final_model_path),
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'val_split': args.val_split
        },
        'data_statistics': {
            'total_samples': len(features),
            'handover_samples': handover_mask.sum().item(),
            'maintain_samples': maintain_mask.sum().item(),
            'a4_events': len(a4_events),
            'd2_events': len(d2_events)
        },
        'evaluation_metrics': {
            'overall_accuracy': accuracy,
            'handover_accuracy': handover_acc,
            'maintain_accuracy': maintain_acc
        }
    }

    results_path = Path('results') / f'bc_evaluation_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n📊 Evaluation Results:")
    logger.info(f"  Overall Accuracy: {accuracy:.2f}%")
    logger.info(f"  Handover Accuracy: {handover_acc:.2f}%")
    logger.info(f"  Maintain Accuracy: {maintain_acc:.2f}%")
    logger.info(f"  Results saved: {results_path}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ Training Complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
