"""
Shaft Point Cloud Segmentation CLI Tool
Command-line interface for processing shaft point clouds with PointNet++ segmentation.

Usage:
    python shaft_segmentation_cli.py --input shaft.las --output output_dir --checkpoint model.pth
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Point cloud libraries
try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

try:
    import pye57
    HAS_E57 = True
except ImportError:
    HAS_E57 = False

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import plyfile
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False


# Class name mapping (from training)
CLASS_NAMES = ['wall', 'pipe', 'guard', 'bunton', 'wire', 'column', 'sheet', 'column2']
CLASS_COLORS = [
    (51, 102, 204),    # Blue - wall
    (255, 153, 0),     # Orange - pipe
    (51, 179, 51),     # Green - guard
    (230, 51, 51),     # Red - bunton (guard2)
    (128, 0, 128),     # Purple - wire
    (0, 204, 204),     # Cyan - column
    (230, 230, 51),    # Yellow - sheet
    (128, 128, 128),   # Gray - column2
]

# Map display names for output files
CLASS_FILE_NAMES = {
    'wall': 'walls',
    'pipe': 'pipes',
    'guard': 'guards',
    'bunton': 'buntons',
    'wire': 'wires',
    'column': 'columns',
    'sheet': 'sheets',
    'column2': 'columns_secondary'
}


def load_point_cloud(file_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load point cloud from various formats.
    Returns: points (N,3), colors (N,3) normalized 0-1, intensities (N,) or None
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext == '.las' or ext == '.laz':
        if not HAS_LASPY:
            raise ImportError("laspy required for LAS files. Install with: pip install laspy")
        las = laspy.read(str(file_path))
        points = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)

        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            colors = np.vstack([las.red, las.green, las.blue]).T.astype(np.float32)
            if colors.max() > 255:
                colors = colors / 65535.0
            else:
                colors = colors / 255.0
        else:
            colors = None

        intensity = np.array(las.intensity).astype(np.float32) if hasattr(las, 'intensity') else None
        return points, colors, intensity

    elif ext == '.e57':
        if not HAS_E57:
            raise ImportError("pye57 required for E57 files. Install with: pip install pye57")
        e57 = pye57.E57(str(file_path))
        data = e57.read_scan(0)
        points = np.column_stack([data['cartesianX'], data['cartesianY'], data['cartesianZ']])

        if 'colorRed' in data:
            colors = np.column_stack([data['colorRed'], data['colorGreen'], data['colorBlue']]) / 255.0
        else:
            colors = None

        intensity = data.get('intensity', None)
        return points, colors, intensity

    elif ext == '.ply':
        if HAS_OPEN3D:
            pcd = o3d.io.read_point_cloud(str(file_path))
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None
            return points, colors, None
        elif HAS_PLYFILE:
            plydata = plyfile.PlyData.read(str(file_path))
            vertex = plydata['vertex']
            points = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
            if 'red' in vertex:
                colors = np.column_stack([vertex['red'], vertex['green'], vertex['blue']]) / 255.0
            else:
                colors = None
            return points, colors, None
        else:
            raise ImportError("open3d or plyfile required for PLY files")

    elif ext == '.pcd':
        if not HAS_OPEN3D:
            raise ImportError("open3d required for PCD files. Install with: pip install open3d")
        pcd = o3d.io.read_point_cloud(str(file_path))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        return points, colors, None

    else:
        raise ValueError(f"Unsupported file format: {ext}")


def save_ply(file_path: str, points: np.ndarray, colors: np.ndarray = None,
             classifications: np.ndarray = None):
    """Save point cloud to PLY format with colors."""
    n_points = len(points)

    with open(file_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        if classifications is not None:
            f.write("property uchar classification\n")
        f.write("end_header\n")

        for i in range(n_points):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
            if colors is not None:
                r, g, b = colors[i]
                if isinstance(r, float) and r <= 1.0:
                    r, g, b = int(r * 255), int(g * 255), int(b * 255)
                line += f" {int(r)} {int(g)} {int(b)}"
            if classifications is not None:
                line += f" {int(classifications[i])}"
            f.write(line + "\n")


class SliceInferenceDataset(Dataset):
    """Dataset for inference on a single slice."""

    def __init__(self, points: np.ndarray, colors: np.ndarray,
                 num_points: int = 4096, block_size: float = 1.0, stride: float = 0.5):
        self.points = points.astype(np.float32)
        self.colors = colors.astype(np.float32) if colors is not None else np.ones((len(points), 3), dtype=np.float32) * 0.148
        self.num_points = num_points
        self.block_size = block_size
        self.stride = stride

        self.blocks = self._create_blocks()

    def _create_blocks(self) -> List[Tuple[float, float, np.ndarray]]:
        blocks = []
        min_bound = self.points.min(axis=0)
        max_bound = self.points.max(axis=0)

        x_range = np.arange(min_bound[0], max_bound[0], self.stride)
        y_range = np.arange(min_bound[1], max_bound[1], self.stride)

        for x in x_range:
            for y in y_range:
                mask = (
                    (self.points[:, 0] >= x) & (self.points[:, 0] < x + self.block_size) &
                    (self.points[:, 1] >= y) & (self.points[:, 1] < y + self.block_size)
                )
                indices = np.where(mask)[0]

                if len(indices) >= 10:
                    blocks.append((x, y, indices))

        return blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x, y, point_indices = self.blocks[idx]

        block_points = self.points[point_indices].copy()
        block_colors = self.colors[point_indices].copy()
        original_indices = point_indices.copy()

        if len(block_points) > self.num_points:
            choice = np.random.choice(len(block_points), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(block_points), self.num_points, replace=True)

        sampled_points = block_points[choice]
        sampled_colors = block_colors[choice]
        sampled_indices = original_indices[choice]

        block_center = np.array([
            x + self.block_size / 2,
            y + self.block_size / 2,
            sampled_points[:, 2].mean()
        ])
        sampled_points = sampled_points - block_center

        return {
            'points': torch.from_numpy(sampled_points).float(),
            'features': torch.from_numpy(sampled_colors).float(),
            'indices': torch.from_numpy(sampled_indices).long(),
            'block_center': torch.from_numpy(block_center).float()
        }


class ShaftSegmentationProcessor:
    """Main processing engine for shaft segmentation."""

    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.config = None
        self.class_names = CLASS_NAMES
        self.num_classes = len(CLASS_NAMES)

    def load_model(self):
        """Load the model from checkpoint."""
        print(f"Loading model from {self.checkpoint_path}...")
        print(f"Using device: {self.device}")

        # Try local models first, then training directory
        try:
            from models.pointnet2 import PointNet2Segmentation, PointNet2SegmentationMSG, PointNet2SegmentationLight
        except ImportError:
            training_dir = Path(self.checkpoint_path).parent.parent
            sys.path.insert(0, str(training_dir))
            from models.pointnet2 import PointNet2Segmentation, PointNet2SegmentationMSG, PointNet2SegmentationLight

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        self.config = checkpoint['config']
        dataset_info = checkpoint['dataset_info']
        self.num_classes = dataset_info['num_classes']
        self.class_names = dataset_info['class_names']

        # Fix class name mapping (guard2 -> bunton)
        self.class_names = [name if name != 'guard2' else 'bunton' for name in self.class_names]

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

    def extract_slices(self, points: np.ndarray, colors: np.ndarray,
                       intensity: np.ndarray, slice_height: float = 10.0) -> List[Dict]:
        """Extract 10m vertical slices from the point cloud."""
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        total_height = z_max - z_min

        print(f"\nPoint cloud height: {total_height:.1f}m (z: {z_min:.1f} to {z_max:.1f})")

        slices = []
        current_z = z_min
        slice_idx = 0

        while current_z < z_max:
            slice_z_max = min(current_z + slice_height, z_max)

            mask = (points[:, 2] >= current_z) & (points[:, 2] < slice_z_max)
            slice_points = points[mask]
            slice_colors = colors[mask] if colors is not None else None
            slice_intensity = intensity[mask] if intensity is not None else None

            if len(slice_points) > 100:
                slices.append({
                    'index': slice_idx,
                    'z_min': current_z,
                    'z_max': slice_z_max,
                    'points': slice_points,
                    'colors': slice_colors,
                    'intensity': slice_intensity,
                    'original_indices': np.where(mask)[0]
                })
                print(f"  Slice {slice_idx}: z=[{current_z:.1f}, {slice_z_max:.1f}], {len(slice_points):,} points")

            slice_idx += 1
            current_z = slice_z_max

        return slices

    @torch.no_grad()
    def process_slice(self, slice_data: Dict, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on a single slice."""
        points = slice_data['points']
        colors = slice_data['colors']
        intensity = slice_data['intensity']

        # Prepare colors
        if colors is not None:
            features = colors.copy()
        elif intensity is not None:
            norm_intensity = intensity / (intensity.mean() + 1e-8) * 0.149
            norm_intensity = np.clip(norm_intensity, 0, 1)
            features = np.column_stack([norm_intensity, norm_intensity, norm_intensity])
        else:
            features = np.ones((len(points), 3), dtype=np.float32) * 0.148

        dataset = SliceInferenceDataset(points, features, num_points=4096, block_size=1.0, stride=0.5)

        if len(dataset) == 0:
            return np.zeros(len(points), dtype=np.int32), np.zeros(len(points), dtype=np.float32)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        num_points = len(points)
        vote_counts = np.zeros((num_points, self.num_classes), dtype=np.float32)

        for batch in loader:
            pts = batch['points'].to(self.device)
            feats = batch['features'].to(self.device)
            indices = batch['indices'].numpy()

            outputs = self.model(pts, feats)
            probs = F.softmax(outputs, dim=-1).cpu().numpy()

            B, N, C = probs.shape
            for b in range(B):
                batch_indices = indices[b]
                batch_probs = probs[b]

                for n in range(N):
                    idx = batch_indices[n]
                    if idx < num_points:
                        vote_counts[idx] += batch_probs[n]

        predictions = vote_counts.argmax(axis=-1).astype(np.int32)
        confidences = vote_counts.max(axis=-1) / (vote_counts.sum(axis=-1) + 1e-8)

        return predictions, confidences

    def create_slice_visualization(self, slice_data: Dict, predictions: np.ndarray,
                                   output_path: str, sample_size: int = 50000):
        """Create visualization for a slice."""
        points = slice_data['points']

        if len(points) > sample_size:
            indices = np.random.choice(len(points), sample_size, replace=False)
            plot_points = points[indices]
            plot_preds = predictions[indices]
        else:
            plot_points = points
            plot_preds = predictions

        colors = np.array(CLASS_COLORS) / 255.0

        fig = plt.figure(figsize=(20, 15))

        ax1 = fig.add_subplot(2, 2, 1)
        for cls in range(self.num_classes):
            mask = plot_preds == cls
            if mask.any():
                name = self.class_names[cls]
                ax1.scatter(plot_points[mask, 0], plot_points[mask, 1],
                           s=0.5, alpha=0.5, c=[colors[cls % len(colors)]],
                           label=f'{name} ({mask.sum():,})')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title(f'Top View (XY) - Slice {slice_data["index"]}')
        ax1.legend(markerscale=10, fontsize=8)
        ax1.set_aspect('equal')

        ax2 = fig.add_subplot(2, 2, 2)
        for cls in range(self.num_classes):
            mask = plot_preds == cls
            if mask.any():
                ax2.scatter(plot_points[mask, 0], plot_points[mask, 2],
                           s=0.5, alpha=0.5, c=[colors[cls % len(colors)]])
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title('Side View (XZ)')

        ax3 = fig.add_subplot(2, 2, 3)
        for cls in range(self.num_classes):
            mask = plot_preds == cls
            if mask.any():
                ax3.scatter(plot_points[mask, 1], plot_points[mask, 2],
                           s=0.5, alpha=0.5, c=[colors[cls % len(colors)]])
        ax3.set_xlabel('Y (m)')
        ax3.set_ylabel('Z (m)')
        ax3.set_title('Side View (YZ)')

        ax4 = fig.add_subplot(2, 2, 4)
        class_counts = [(predictions == i).sum() for i in range(self.num_classes)]
        ax4.bar(self.class_names, class_counts,
                color=[colors[i % len(colors)] for i in range(self.num_classes)])
        ax4.set_xlabel('Class')
        ax4.set_ylabel('Point Count')
        ax4.set_title('Class Distribution')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def process_shaft(input_path: str, output_dir: str, checkpoint_path: str,
                  slice_height: float = 10.0, device: str = None, batch_size: int = 32):
    """Main processing function."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = output_dir / "visualizations"
    families_dir = output_dir / "classified_families"
    vis_dir.mkdir(exist_ok=True)
    families_dir.mkdir(exist_ok=True)

    # Initialize processor
    processor = ShaftSegmentationProcessor(checkpoint_path, device)
    processor.load_model()

    # Load point cloud
    print(f"\nLoading point cloud: {input_path}")
    points, colors, intensity = load_point_cloud(input_path)
    print(f"Loaded {len(points):,} points")

    # Prepare colors
    if colors is None and intensity is not None:
        norm_intensity = intensity / (intensity.mean() + 1e-8) * 0.149
        norm_intensity = np.clip(norm_intensity, 0, 1)
        colors = np.column_stack([norm_intensity, norm_intensity, norm_intensity])
    elif colors is None:
        colors = np.ones((len(points), 3), dtype=np.float32) * 0.148

    # Extract slices
    print("\nExtracting vertical slices...")
    slices = processor.extract_slices(points, colors, intensity, slice_height)
    print(f"Extracted {len(slices)} slices")

    # Initialize predictions
    all_predictions = np.zeros(len(points), dtype=np.int32)
    all_confidences = np.zeros(len(points), dtype=np.float32)

    # Process each slice
    for i, slice_data in enumerate(slices):
        print(f"\n{'='*60}")
        print(f"Processing slice {i+1}/{len(slices)} (z={slice_data['z_min']:.1f}m to {slice_data['z_max']:.1f}m)")

        predictions, confidences = processor.process_slice(slice_data, batch_size)

        original_indices = slice_data['original_indices']
        all_predictions[original_indices] = predictions
        all_confidences[original_indices] = confidences

        # Print class distribution
        print("Class distribution:")
        for cls in range(processor.num_classes):
            count = (predictions == cls).sum()
            name = processor.class_names[cls]
            print(f"  {name}: {count:,} points ({100*count/len(predictions):.1f}%)")

        # Create visualization
        vis_path = vis_dir / f"slice_{i:03d}_z{slice_data['z_min']:.0f}_{slice_data['z_max']:.0f}.png"
        processor.create_slice_visualization(slice_data, predictions, str(vis_path))
        print(f"Saved visualization: {vis_path.name}")

    # Generate classified colors
    print(f"\n{'='*60}")
    print("Generating output files...")

    classified_colors = np.zeros((len(points), 3), dtype=np.uint8)
    for cls in range(processor.num_classes):
        mask = all_predictions == cls
        classified_colors[mask] = CLASS_COLORS[cls]

    # Save combined PLY
    combined_ply_path = output_dir / "combined_classified.ply"
    save_ply(str(combined_ply_path), points, classified_colors / 255.0, all_predictions)
    print(f"Saved combined classified PLY: {combined_ply_path}")

    # Save per-class PLY files
    print("\nSaving per-class PLY files...")
    for cls in range(processor.num_classes):
        mask = all_predictions == cls
        if mask.sum() > 0:
            class_name = processor.class_names[cls]
            file_name = CLASS_FILE_NAMES.get(class_name, class_name)
            class_points = points[mask]
            class_colors = classified_colors[mask]

            class_ply_path = families_dir / f"{file_name}.ply"
            save_ply(str(class_ply_path), class_points, class_colors / 255.0)
            print(f"  Saved {file_name}.ply ({mask.sum():,} points)")

    # Save metadata
    metadata = {
        'input_file': str(input_path),
        'total_points': len(points),
        'num_slices': len(slices),
        'slice_height_m': slice_height,
        'class_distribution': {
            processor.class_names[i]: int((all_predictions == i).sum())
            for i in range(processor.num_classes)
        },
        'mean_confidence': float(all_confidences.mean()),
        'processing_date': datetime.now().isoformat()
    }

    with open(output_dir / "processing_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Shaft Point Cloud Segmentation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python shaft_segmentation_cli.py --input shaft.las --output results/ --checkpoint model.pth
  python shaft_segmentation_cli.py --input shaft.e57 --output results/ --slice-height 5.0
        """
    )

    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input point cloud file (LAS, E57, PLY, PCD)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Model checkpoint path (default: looks for checkpoints_best/latest.pth)')
    parser.add_argument('--slice-height', type=float, default=10.0,
                        help='Height of each slice in meters (default: 10.0)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, default: auto)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference (default: 32)')

    args = parser.parse_args()

    # Find checkpoint if not specified
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # Try to find checkpoint relative to script location
        script_dir = Path(__file__).parent
        default_paths = [
            script_dir / "checkpoints" / "latest.pth",
            script_dir.parent / "training" / "checkpoints_best" / "latest.pth",
            Path("checkpoints_best/latest.pth"),
        ]
        for path in default_paths:
            if path.exists():
                checkpoint_path = str(path)
                print(f"Using checkpoint: {checkpoint_path}")
                break

        if checkpoint_path is None:
            print("ERROR: No checkpoint found. Please specify with --checkpoint")
            sys.exit(1)

    # Validate inputs
    if not Path(args.input).exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Run processing
    process_shaft(
        input_path=args.input,
        output_dir=args.output,
        checkpoint_path=checkpoint_path,
        slice_height=args.slice_height,
        device=args.device,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
