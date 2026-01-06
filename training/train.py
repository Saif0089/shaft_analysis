"""
Training script for PointNet++ shaft segmentation
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.pointnet2 import PointNet2Segmentation, PointNet2SegmentationMSG, PointNet2SegmentationLight
from data.dataset import create_dataloaders


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, ignore_index: int = -1):
        super().__init__()
        self.alpha = alpha  # Per-class weights
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            inputs.reshape(-1, inputs.size(-1)),
            targets.reshape(-1),
            weight=self.alpha,
            reduction='none',
            ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""

    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (B, N, C), targets: (B, N)
        B, N, C = inputs.shape
        inputs = inputs.reshape(-1, C)
        targets = targets.reshape(-1)

        # Mask out ignore index
        mask = targets != self.ignore_index
        inputs = inputs[mask]
        targets = targets[mask]

        # Softmax
        probs = torch.softmax(inputs, dim=-1)
        targets_onehot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1)

        # Dice score per class
        intersection = (probs * targets_onehot).sum(dim=0)
        union = probs.sum(dim=0) + targets_onehot.sum(dim=0)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined Cross-Entropy + Dice Loss"""

    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5,
                 class_weights: Optional[torch.Tensor] = None, ignore_index: int = -1):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(inputs.reshape(-1, inputs.size(-1)), targets.reshape(-1))
        dice = self.dice_loss(inputs, targets)
        return self.ce_weight * ce + self.dice_weight * dice


class FocalDiceLoss(nn.Module):
    """Combined Focal + Dice Loss for better class imbalance handling"""

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0,
                 focal_weight: float = 0.5, dice_weight: float = 0.5, ignore_index: int = -1):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.focal_weight * focal + self.dice_weight * dice


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
                    ignore_index: int = -1) -> Dict[str, float]:
    """Compute segmentation metrics"""
    pred = pred.view(-1)
    target = target.view(-1)

    # Mask out ignore index
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]

    # Overall accuracy
    correct = (pred == target).sum().item()
    total = target.numel()
    accuracy = correct / total if total > 0 else 0

    # Per-class IoU
    ious = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union > 0:
            ious.append(intersection / union)

    mean_iou = np.mean(ious) if ious else 0

    return {
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'per_class_iou': ious
    }


class Trainer:
    """Training manager for PointNet++"""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        dataset_info: Dict,
        config: Dict,
        device: torch.device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dataset_info = dataset_info
        self.config = config
        self.device = device

        # Loss function - use Focal Loss for better handling of class imbalance
        class_weights = self._compute_class_weights() if config.get('use_class_weights', True) else None

        loss_type = config.get('loss_type', 'focal_dice')
        if loss_type == 'focal_dice':
            self.criterion = FocalDiceLoss(
                alpha=class_weights,
                gamma=config.get('focal_gamma', 2.0),
                focal_weight=0.5,
                dice_weight=0.5,
                ignore_index=-1
            )
            print(f"Using Focal+Dice Loss (gamma={config.get('focal_gamma', 2.0)})")
        elif loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=class_weights,
                gamma=config.get('focal_gamma', 2.0),
                ignore_index=-1
            )
            print(f"Using Focal Loss (gamma={config.get('focal_gamma', 2.0)})")
        else:
            self.criterion = CombinedLoss(
                ce_weight=config.get('ce_weight', 0.5),
                dice_weight=config.get('dice_weight', 0.5),
                class_weights=class_weights
            )

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )

        # Learning rate scheduler with warmup
        warmup_epochs = config.get('warmup_epochs', 10)
        self.warmup_epochs = warmup_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 100) - warmup_epochs,
            eta_min=config.get('min_lr', 1e-6)
        )
        print(f"Using LR warmup for {warmup_epochs} epochs")

        # Logging
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(config.get('log_dir', 'logs')) / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

        # Best metrics
        self.best_iou = 0
        self.best_epoch = 0

        # Save config
        with open(self.log_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        with open(self.log_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)

    def _compute_class_weights(self) -> torch.Tensor:
        """Compute class weights based on frequency - inverse frequency weighting"""
        # Count labels in training data
        label_counts = {}
        for batch in self.train_loader:
            labels = batch['labels'].numpy().flatten()
            for l in labels:
                if l >= 0:  # Ignore -1 (unlabeled)
                    label_counts[l] = label_counts.get(l, 0) + 1

        num_classes = self.dataset_info['num_classes']
        weights = torch.ones(num_classes)

        if label_counts:
            total = sum(label_counts.values())
            max_count = max(label_counts.values())

            # Use stronger inverse frequency weighting for rare classes
            for cls, count in label_counts.items():
                if cls < num_classes:
                    # More aggressive: inverse frequency (not sqrt)
                    freq = count / total
                    # Cap the weight to prevent instability
                    weights[cls] = min(1.0 / (freq + 1e-6), 20.0)

            # Normalize so average weight is 1
            weights = weights / weights.mean()

            # Print class distribution
            print(f"Class distribution in training:")
            for cls, count in sorted(label_counts.items()):
                name = self.dataset_info['class_names'][cls] if cls < len(self.dataset_info['class_names']) else f'class_{cls}'
                print(f"  {name}: {count:,} ({100*count/total:.1f}%) -> weight {weights[cls]:.2f}")

        print(f"Class weights: {[f'{w:.2f}' for w in weights.tolist()]}")
        return weights.to(self.device)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []

        for batch_idx, batch in enumerate(self.train_loader):
            points = batch['points'].to(self.device)
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(points, features if features.size(-1) > 0 else None)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

            # Collect predictions
            pred = outputs.argmax(dim=-1)
            all_preds.append(pred.cpu())
            all_targets.append(labels.cpu())

            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')

        # Compute metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = compute_metrics(all_preds, all_targets, self.dataset_info['num_classes'])
        metrics['loss'] = total_loss / len(self.train_loader)

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        for batch in self.val_loader:
            points = batch['points'].to(self.device)
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(points, features if features.size(-1) > 0 else None)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()

            pred = outputs.argmax(dim=-1)
            all_preds.append(pred.cpu())
            all_targets.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = compute_metrics(all_preds, all_targets, self.dataset_info['num_classes'])
        metrics['loss'] = total_loss / len(self.val_loader)

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'dataset_info': self.dataset_info
        }

        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')

        # Save periodic
        if epoch % self.config.get('save_every', 10) == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'epoch_{epoch:03d}.pth')

        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')

    def train(self):
        """Full training loop"""
        epochs = self.config.get('epochs', 100)
        print(f"\nStarting training for {epochs} epochs")
        print(f"Classes: {self.dataset_info['class_names']}")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        print(f"Checkpoints: {self.checkpoint_dir}")
        print(f"Logs: {self.log_dir}\n")

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"mIoU: {train_metrics['mean_iou']:.4f}")

            # Validate
            val_metrics = self.validate()
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"mIoU: {val_metrics['mean_iou']:.4f}")

            # Update learning rate with warmup
            if epoch <= self.warmup_epochs:
                # Linear warmup
                warmup_factor = epoch / self.warmup_epochs
                current_lr = self.config.get('lr', 1e-3) * warmup_factor
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
            else:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]

            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('mIoU/train', train_metrics['mean_iou'], epoch)
            self.writer.add_scalar('mIoU/val', val_metrics['mean_iou'], epoch)
            self.writer.add_scalar('LR', current_lr, epoch)

            # Check for best
            is_best = val_metrics['mean_iou'] > self.best_iou
            if is_best:
                self.best_iou = val_metrics['mean_iou']
                self.best_epoch = epoch
                print(f"  New best mIoU: {self.best_iou:.4f}")

            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)

            elapsed = time.time() - start_time
            print(f"  Time: {elapsed:.1f}s, LR: {current_lr:.2e}\n")

        print(f"\nTraining complete!")
        print(f"Best mIoU: {self.best_iou:.4f} at epoch {self.best_epoch}")
        print(f"Model saved to: {self.checkpoint_dir / 'best.pth'}")

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train PointNet++ for shaft segmentation')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to annotation exports')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_points', type=int, default=4096, help='Points per block')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--block_size', type=float, default=1.0, help='Block size in meters')
    parser.add_argument('--stride', type=float, default=0.5, help='Stride between blocks')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='Tensorboard log directory')
    parser.add_argument('--use_msg', action='store_true', help='Use multi-scale grouping')
    parser.add_argument('--light', action='store_true', help='Use lightweight model (fewer params, better for small datasets)')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create dataloaders
    train_loader, val_loader, dataset_info = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_points=args.num_points,
        block_size=args.block_size,
        stride=args.stride,
        num_workers=4
    )

    # Create model
    num_classes = dataset_info['num_classes']
    in_channels = 3  # RGB

    if args.light:
        model = PointNet2SegmentationLight(num_classes=num_classes, in_channels=in_channels)
        model_name = 'PointNet++ Light'
    elif args.use_msg:
        model = PointNet2SegmentationMSG(num_classes=num_classes, in_channels=in_channels)
        model_name = 'PointNet++ MSG'
    else:
        model = PointNet2Segmentation(num_classes=num_classes, in_channels=in_channels)
        model_name = 'PointNet++'

    print(f"\nModel: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Config
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'num_points': args.num_points,
        'lr': args.lr,
        'min_lr': 1e-6,
        'weight_decay': 1e-4,
        'block_size': args.block_size,
        'stride': args.stride,
        'use_msg': args.use_msg,
        'use_light': args.light,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'loss_type': 'focal_dice',
        'focal_gamma': args.focal_gamma,
        'ce_weight': 0.5,
        'dice_weight': 0.5,
        'use_class_weights': True,
        'warmup_epochs': 10,
        'save_every': 10
    }

    # Resume from checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resumed from: {args.resume}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset_info=dataset_info,
        config=config,
        device=device
    )

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
