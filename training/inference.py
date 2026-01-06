"""
Inference script for PointNet++ shaft segmentation
Runs prediction on point clouds and saves results
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import laspy
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from models.pointnet2 import PointNet2Segmentation, PointNet2SegmentationMSG, PointNet2SegmentationLight
from data.dataset import ShaftInferenceDataset


class PointCloudPredictor:
    """Inference engine for point cloud segmentation"""

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load checkpoint
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.config = checkpoint['config']
        self.dataset_info = checkpoint['dataset_info']
        self.num_classes = self.dataset_info['num_classes']
        self.class_names = self.dataset_info['class_names']

        # Create model
        if self.config.get('use_light', False):
            self.model = PointNet2SegmentationLight(num_classes=self.num_classes, in_channels=3)
        elif self.config.get('use_msg', False):
            self.model = PointNet2SegmentationMSG(num_classes=self.num_classes, in_channels=3)
        else:
            self.model = PointNet2Segmentation(num_classes=self.num_classes, in_channels=3)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded. Classes: {self.class_names}")
        print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Metrics: {checkpoint.get('metrics', {})}")

    @torch.no_grad()
    def predict(
        self,
        las_path: str,
        num_points: int = 4096,
        block_size: float = 1.0,
        stride: float = 0.5,
        batch_size: int = 16,
        voting: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run prediction on a point cloud file

        Args:
            las_path: Path to LAS file
            num_points: Points per block
            block_size: Block size in meters
            stride: Stride between blocks
            batch_size: Batch size for inference
            voting: Use voting for overlapping predictions

        Returns:
            points: Original point coordinates
            predictions: Per-point class predictions
            confidences: Per-point prediction confidence
        """
        print(f"\nRunning inference on {las_path}")

        # Create dataset
        dataset = ShaftInferenceDataset(
            las_path=las_path,
            num_points=num_points,
            block_size=block_size,
            stride=stride,
            use_colors=True
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Initialize prediction arrays
        num_total_points = len(dataset.original_points)
        if voting:
            # Accumulate votes for each point
            vote_counts = np.zeros((num_total_points, self.num_classes), dtype=np.float32)
        else:
            predictions = np.full(num_total_points, -1, dtype=np.int32)
            confidences = np.zeros(num_total_points, dtype=np.float32)

        print(f"Processing {len(loader)} batches...")

        for batch_idx, batch in enumerate(loader):
            points = batch['points'].to(self.device)
            features = batch['features'].to(self.device)
            indices = batch['indices'].numpy()

            # Forward pass
            outputs = self.model(points, features)
            probs = F.softmax(outputs, dim=-1).cpu().numpy()

            # Aggregate predictions
            B, N, C = probs.shape
            for b in range(B):
                batch_indices = indices[b]
                batch_probs = probs[b]

                if voting:
                    # Add votes
                    for n in range(N):
                        idx = batch_indices[n]
                        if idx < num_total_points:
                            vote_counts[idx] += batch_probs[n]
                else:
                    # Use highest confidence
                    pred = batch_probs.argmax(axis=-1)
                    conf = batch_probs.max(axis=-1)

                    for n in range(N):
                        idx = batch_indices[n]
                        if idx < num_total_points and conf[n] > confidences[idx]:
                            predictions[idx] = pred[n]
                            confidences[idx] = conf[n]

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(loader)}")

        # Finalize predictions
        if voting:
            predictions = vote_counts.argmax(axis=-1).astype(np.int32)
            confidences = vote_counts.max(axis=-1) / (vote_counts.sum(axis=-1) + 1e-8)
        else:
            # Handle unpredicted points
            unpredicted = predictions == -1
            if unpredicted.any():
                print(f"  {unpredicted.sum()} points had no predictions, assigning to class 0")
                predictions[unpredicted] = 0
                confidences[unpredicted] = 0

        print(f"Prediction complete. Class distribution:")
        for cls in range(self.num_classes):
            count = (predictions == cls).sum()
            name = self.class_names[cls] if cls < len(self.class_names) else f'class_{cls}'
            print(f"  {name}: {count:,} points ({100 * count / len(predictions):.1f}%)")

        return dataset.original_points, predictions, confidences

    def save_results(
        self,
        output_path: str,
        points: np.ndarray,
        predictions: np.ndarray,
        confidences: np.ndarray,
        original_las_path: Optional[str] = None
    ):
        """Save prediction results to LAS file"""
        print(f"\nSaving results to {output_path}")

        # Create output LAS
        if original_las_path:
            # Copy from original
            original = laspy.read(original_las_path)
            output = laspy.create(point_format=original.point_format, file_version=original.header.version)
            output.x = original.x
            output.y = original.y
            output.z = original.z

            if hasattr(original, 'intensity'):
                output.intensity = original.intensity
        else:
            # Create new
            output = laspy.create(point_format=0, file_version="1.2")
            output.x = points[:, 0]
            output.y = points[:, 1]
            output.z = points[:, 2]

        # Store predictions in classification field
        output.classification = predictions.astype(np.uint8)

        # Color by class
        colors = self._get_class_colors()
        red = np.zeros(len(points), dtype=np.uint16)
        green = np.zeros(len(points), dtype=np.uint16)
        blue = np.zeros(len(points), dtype=np.uint16)

        for cls in range(self.num_classes):
            mask = predictions == cls
            color = colors[cls % len(colors)]
            red[mask] = int(color[0] * 65535)
            green[mask] = int(color[1] * 65535)
            blue[mask] = int(color[2] * 65535)

        if hasattr(output, 'red'):
            output.red = red
            output.green = green
            output.blue = blue

        output.write(output_path)
        print(f"Saved {len(points):,} points")

        # Also save metadata
        meta_path = Path(output_path).with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'num_points': len(points),
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'class_distribution': {
                    self.class_names[i] if i < len(self.class_names) else f'class_{i}':
                    int((predictions == i).sum())
                    for i in range(self.num_classes)
                },
                'mean_confidence': float(confidences.mean())
            }, f, indent=2)

        print(f"Saved metadata to {meta_path}")

    def _get_class_colors(self) -> List[Tuple[float, float, float]]:
        """Get colors for each class"""
        return [
            (0.2, 0.4, 0.8),   # Blue
            (1.0, 0.6, 0.0),   # Orange
            (0.2, 0.7, 0.2),   # Green
            (0.9, 0.2, 0.2),   # Red
            (0.5, 0.0, 0.5),   # Purple
            (0.0, 0.8, 0.8),   # Cyan
            (0.9, 0.9, 0.2),   # Yellow
            (0.5, 0.5, 0.5),   # Gray
            (1.0, 0.4, 0.7),   # Pink
            (0.0, 0.5, 0.0),   # Dark green
        ]

    def visualize_results(
        self,
        points: np.ndarray,
        predictions: np.ndarray,
        output_path: str,
        sample_size: int = 50000
    ):
        """Create visualization of prediction results"""
        print(f"\nCreating visualization...")

        # Sample points for plotting
        if len(points) > sample_size:
            indices = np.random.choice(len(points), sample_size, replace=False)
            plot_points = points[indices]
            plot_preds = predictions[indices]
        else:
            plot_points = points
            plot_preds = predictions

        colors = self._get_class_colors()

        fig = plt.figure(figsize=(20, 15))

        # Top view
        ax1 = fig.add_subplot(2, 2, 1)
        for cls in range(self.num_classes):
            mask = plot_preds == cls
            if mask.any():
                name = self.class_names[cls] if cls < len(self.class_names) else f'class_{cls}'
                ax1.scatter(
                    plot_points[mask, 0], plot_points[mask, 1],
                    s=0.5, alpha=0.5, c=[colors[cls % len(colors)]],
                    label=f'{name} ({mask.sum():,})'
                )
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Top View (XY)')
        ax1.legend(markerscale=10, fontsize=8)
        ax1.set_aspect('equal')

        # Side view XZ
        ax2 = fig.add_subplot(2, 2, 2)
        for cls in range(self.num_classes):
            mask = plot_preds == cls
            if mask.any():
                ax2.scatter(
                    plot_points[mask, 0], plot_points[mask, 2],
                    s=0.5, alpha=0.5, c=[colors[cls % len(colors)]]
                )
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title('Side View (XZ)')

        # Side view YZ
        ax3 = fig.add_subplot(2, 2, 3)
        for cls in range(self.num_classes):
            mask = plot_preds == cls
            if mask.any():
                ax3.scatter(
                    plot_points[mask, 1], plot_points[mask, 2],
                    s=0.5, alpha=0.5, c=[colors[cls % len(colors)]]
                )
        ax3.set_xlabel('Y (m)')
        ax3.set_ylabel('Z (m)')
        ax3.set_title('Side View (YZ)')

        # Class distribution
        ax4 = fig.add_subplot(2, 2, 4)
        class_counts = [(predictions == i).sum() for i in range(self.num_classes)]
        class_labels = [self.class_names[i] if i < len(self.class_names) else f'class_{i}'
                        for i in range(self.num_classes)]
        bars = ax4.bar(class_labels, class_counts, color=[colors[i % len(colors)] for i in range(self.num_classes)])
        ax4.set_xlabel('Class')
        ax4.set_ylabel('Point Count')
        ax4.set_title('Class Distribution')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run inference on point clouds')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input LAS file')
    parser.add_argument('--output', type=str, default=None, help='Output LAS file')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_points', type=int, default=4096, help='Points per block')
    parser.add_argument('--block_size', type=float, default=1.0, help='Block size in meters')
    parser.add_argument('--stride', type=float, default=0.5, help='Stride between blocks')
    parser.add_argument('--no_voting', action='store_true', help='Disable voting for overlapping blocks')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output path
    if args.output:
        output_path = args.output
    else:
        input_stem = Path(args.input).stem
        output_path = output_dir / f'{input_stem}_predicted.las'

    # Create predictor
    predictor = PointCloudPredictor(args.checkpoint)

    # Run prediction
    points, predictions, confidences = predictor.predict(
        las_path=args.input,
        num_points=args.num_points,
        block_size=args.block_size,
        stride=args.stride,
        batch_size=args.batch_size,
        voting=not args.no_voting
    )

    # Save results
    predictor.save_results(
        output_path=str(output_path),
        points=points,
        predictions=predictions,
        confidences=confidences,
        original_las_path=args.input
    )

    # Visualize
    if args.visualize:
        vis_path = output_dir / f'{Path(args.input).stem}_visualization.png'
        predictor.visualize_results(points, predictions, str(vis_path))


if __name__ == '__main__':
    main()
