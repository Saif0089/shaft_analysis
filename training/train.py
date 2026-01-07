"""Training script for PointNet++ semantic segmentation."""
import os
import sys
import json
import time
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GROUND_TRUTH_DIR, MODELS_DIR, ANNOTATIONS_DIR
from training.model import PointNet2SemSeg
from training.dataset import ShaftPointCloudDataset


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for points, labels in pbar:
        points = points.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward
        outputs = model(points)  # (B, N, num_classes)
        outputs = outputs.view(-1, outputs.shape[-1])  # (B*N, num_classes)
        labels = labels.view(-1)  # (B*N,)

        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100. * correct / total
    }


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
             device: torch.device) -> dict:
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for points, labels in tqdm(dataloader, desc="Validating"):
            points = points.to(device)
            labels = labels.to(device)

            outputs = model(points)
            outputs = outputs.view(-1, outputs.shape[-1])
            labels_flat = labels.view(-1)

            loss = criterion(outputs, labels_flat)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels_flat.size(0)
            correct += predicted.eq(labels_flat).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels_flat.cpu().numpy())

    # Per-class accuracy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    unique_labels = np.unique(all_labels)
    per_class_acc = {}
    for label in unique_labels:
        mask = all_labels == label
        if mask.sum() > 0:
            per_class_acc[int(label)] = (all_predictions[mask] == all_labels[mask]).mean() * 100

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100. * correct / total,
        'per_class_accuracy': per_class_acc
    }


def get_num_classes() -> int:
    """Get number of classes from labels file."""
    labels_file = os.path.join(ANNOTATIONS_DIR, 'labels.json')
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            data = json.load(f)
            return max(c['id'] for c in data['classes']) + 1
    return 2  # Default: unlabeled + 1 class


def train(gt_files: list = None, epochs: int = 100, batch_size: int = 16,
          lr: float = 0.001, num_points: int = 8192, resume: str = None,
          device_id: int = 0) -> str:
    """
    Main training function.

    Args:
        gt_files: List of ground truth LAS files (if None, use all in GROUND_TRUTH_DIR)
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        num_points: Points per sample
        resume: Path to checkpoint to resume from
        device_id: CUDA device ID

    Returns:
        Path to best model checkpoint
    """
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get ground truth files
    if gt_files is None:
        gt_files = [os.path.join(GROUND_TRUTH_DIR, f)
                    for f in os.listdir(GROUND_TRUTH_DIR) if f.endswith('_gt.las')]

    if not gt_files:
        raise ValueError("No ground truth files found!")

    print(f"Training on {len(gt_files)} ground truth files:")
    for f in gt_files:
        print(f"  - {os.path.basename(f)}")

    # Create dataset
    dataset = ShaftPointCloudDataset(gt_files, num_points=num_points, augment=True)

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # For small datasets, use the same data for both
    if len(dataset) < 10:
        train_dataset = dataset
        val_dataset = dataset
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model
    num_classes = get_num_classes()
    print(f"Number of classes: {num_classes}")

    model = PointNet2SemSeg(num_classes=num_classes, in_channels=3).to(device)

    # Loss with class weights
    class_weights = dataset.get_class_weights().to(device)
    print(f"Class weights: {class_weights}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Resume from checkpoint
    start_epoch = 0
    best_accuracy = 0

    if resume and os.path.exists(resume):
        print(f"Resuming from {resume}")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint.get('best_accuracy', 0)

    # Create output directory
    os.makedirs(MODELS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_model_path = os.path.join(MODELS_DIR, f'best_model_{timestamp}.pth')

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    training_log = []

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Log
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'per_class_accuracy': val_metrics['per_class_accuracy'],
            'lr': optimizer.param_groups[0]['lr']
        }
        training_log.append(log_entry)

        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
        print(f"Per-class accuracy: {val_metrics['per_class_accuracy']}")

        # Save best model
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'num_classes': num_classes,
                'training_log': training_log
            }, best_model_path)
            print(f"Saved best model with accuracy {best_accuracy:.2f}%")

        # Early stopping if we reach very high accuracy
        if val_metrics['accuracy'] >= 99.5:
            print("Reached 99.5% accuracy, stopping early.")
            break

    # Save final model
    final_model_path = os.path.join(MODELS_DIR, f'final_model_{timestamp}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_accuracy,
        'num_classes': num_classes,
        'training_log': training_log
    }, final_model_path)

    # Save training log
    log_path = os.path.join(MODELS_DIR, f'training_log_{timestamp}.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best model: {best_model_path}")
    print(f"Best accuracy: {best_accuracy:.2f}%")

    return best_model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PointNet++ for shaft segmentation')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_points', type=int, default=8192, help='Points per sample')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID')

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_points=args.num_points,
        resume=args.resume,
        device_id=args.device
    )
