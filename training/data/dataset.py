"""
Dataset classes for PointNet++ training on shaft segmentation data
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import laspy


class ShaftSegmentationDataset(Dataset):
    """
    Dataset for shaft point cloud segmentation.
    Loads from annotation tool export format or directly from LAS files with labels.
    """

    def __init__(
        self,
        data_dir: str,
        num_points: int = 4096,
        split: str = 'train',
        train_ratio: float = 0.8,
        use_colors: bool = True,
        use_normals: bool = False,
        augment: bool = True,
        block_size: float = 1.0,
        stride: float = 0.5,
        random_seed: int = 42
    ):
        """
        Args:
            data_dir: Directory containing exported annotation data or LAS files
            num_points: Number of points to sample per block
            split: 'train' or 'val'
            train_ratio: Ratio of data for training
            use_colors: Whether to use RGB colors as features
            use_normals: Whether to compute and use normals
            augment: Whether to apply data augmentation
            block_size: Size of blocks to sample (meters)
            stride: Stride between blocks (meters)
            random_seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.num_points = num_points
        self.split = split
        self.use_colors = use_colors
        self.use_normals = use_normals
        self.augment = augment and split == 'train'
        self.block_size = block_size
        self.stride = stride

        np.random.seed(random_seed)

        # Load data
        self.points, self.colors, self.labels, self.label_map = self._load_data()

        # Create blocks
        self.blocks = self._create_blocks()

        # Split into train/val
        n_blocks = len(self.blocks)
        indices = np.random.permutation(n_blocks)
        split_idx = int(n_blocks * train_ratio)

        if split == 'train':
            self.block_indices = indices[:split_idx]
        else:
            self.block_indices = indices[split_idx:]

        print(f"Dataset [{split}]: {len(self.block_indices)} blocks, {len(self.label_map)} classes")

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Load point cloud data from various formats"""

        # Check if data_dir itself is an export folder (has _points.txt files)
        points_files = list(self.data_dir.glob('*_points.txt'))
        if points_files:
            return self._load_from_single_export(self.data_dir)

        # Try loading from annotation export format first
        points_file = self.data_dir / 'pointnet_format'
        if points_file.exists():
            return self._load_from_export(points_file)

        # Try loading from single export folder
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith('export_'):
                return self._load_from_single_export(subdir)

        # Try loading from LAS file with classification
        las_files = list(self.data_dir.glob('*.las'))
        if las_files:
            return self._load_from_las(las_files[0])

        raise FileNotFoundError(f"No valid data found in {self.data_dir}")

    def _load_from_export(self, export_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Load from annotation tool export format"""
        # Find the latest export
        export_folders = sorted([d for d in export_dir.iterdir() if d.is_dir()])
        if not export_folders:
            raise FileNotFoundError(f"No export folders in {export_dir}")

        latest_export = export_folders[-1]
        return self._load_from_single_export(latest_export)

    def _load_from_single_export(self, export_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Load from a single export folder"""
        # Find files
        points_files = list(export_dir.glob('*_points.txt'))
        labels_files = list(export_dir.glob('*_labels.txt'))
        map_files = list(export_dir.glob('*_label_map.json'))

        if not (points_files and labels_files and map_files):
            raise FileNotFoundError(f"Missing files in {export_dir}")

        # Load points (x, y, z, r, g, b)
        point_data = np.loadtxt(points_files[0])
        points = point_data[:, :3]
        colors = point_data[:, 3:6] / 255.0 if point_data.shape[1] >= 6 else np.ones((len(points), 3)) * 0.5

        # Load labels
        labels = np.loadtxt(labels_files[0], dtype=np.int32)

        # Load label map
        with open(map_files[0], 'r') as f:
            label_data = json.load(f)
            label_map = label_data.get('label_to_id', {})

        # Handle background/unlabeled points (-1 labels)
        # Keep original class IDs (0-6) and map -1 to a separate background class
        has_background = -1 in labels

        if has_background:
            # Map -1 to the next available class ID (after the max existing class)
            max_class = max(label_map.values())
            background_id = max_class + 1
            labels = np.where(labels == -1, background_id, labels)
            label_map['background'] = background_id

        # Create new label map preserving original order
        new_label_map = label_map.copy()

        print(f"Loaded {len(points)} points, {len(new_label_map)} classes from {export_dir.name}")
        return points.astype(np.float32), colors.astype(np.float32), labels, new_label_map

    def _load_from_las(self, las_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Load from LAS file with classification field"""
        las = laspy.read(las_path)
        points = np.vstack([las.x, las.y, las.z]).T

        # Get colors
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            colors = np.vstack([las.red, las.green, las.blue]).T / 65535.0
        else:
            colors = np.ones((len(points), 3)) * 0.5

        # Get labels from classification
        labels = las.classification.astype(np.int32)

        # Create label map
        unique_labels = np.unique(labels)
        label_map = {f'class_{i}': i for i in unique_labels}

        print(f"Loaded {len(points)} points from {las_path.name}")
        return points.astype(np.float32), colors.astype(np.float32), labels, label_map

    def _create_blocks(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create blocks of points for training with class-aware sampling"""
        blocks = []
        block_class_counts = []

        # Get point cloud bounds
        min_bound = self.points.min(axis=0)
        max_bound = self.points.max(axis=0)

        # Create grid of block centers
        x_range = np.arange(min_bound[0], max_bound[0], self.stride)
        y_range = np.arange(min_bound[1], max_bound[1], self.stride)

        for x in x_range:
            for y in y_range:
                # Find points in this block
                mask = (
                    (self.points[:, 0] >= x) & (self.points[:, 0] < x + self.block_size) &
                    (self.points[:, 1] >= y) & (self.points[:, 1] < y + self.block_size)
                )
                indices = np.where(mask)[0]

                # Skip blocks with too few points
                if len(indices) < self.num_points // 4:
                    continue

                # Store block info and track which classes are present
                blocks.append((x, y, indices))
                block_labels = self.labels[indices]
                unique_labels = np.unique(block_labels)
                block_class_counts.append(set(unique_labels.tolist()))

        # For training, duplicate blocks containing rare classes
        if self.augment and len(blocks) > 0:
            # Count class frequencies across blocks
            class_block_counts = {}
            for classes in block_class_counts:
                for c in classes:
                    class_block_counts[c] = class_block_counts.get(c, 0) + 1

            # Find rare classes (appear in fewer than 20% of blocks)
            max_count = max(class_block_counts.values())
            rare_classes = {c for c, count in class_block_counts.items() if count < max_count * 0.3}

            if rare_classes:
                print(f"Rare classes (will oversample): {rare_classes}")
                # Oversample blocks with rare classes
                oversampled_blocks = []
                for i, (block, classes) in enumerate(zip(blocks, block_class_counts)):
                    oversampled_blocks.append(block)
                    # Add extra copies of blocks with rare classes
                    if classes & rare_classes:
                        oversample_factor = 2  # Add 2x copies
                        for _ in range(oversample_factor):
                            oversampled_blocks.append(block)
                blocks = oversampled_blocks

        print(f"Created {len(blocks)} blocks of size {self.block_size}m")
        return blocks

    def __len__(self) -> int:
        return len(self.block_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        block_idx = self.block_indices[idx]
        x, y, point_indices = self.blocks[block_idx]

        # Get points in block
        block_points = self.points[point_indices].copy()
        block_colors = self.colors[point_indices].copy()
        block_labels = self.labels[point_indices].copy()

        # Sample or pad to fixed size
        if len(block_points) > self.num_points:
            choice = np.random.choice(len(block_points), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(block_points), self.num_points, replace=True)

        block_points = block_points[choice]
        block_colors = block_colors[choice]
        block_labels = block_labels[choice]

        # Normalize coordinates to block center
        block_center = np.array([x + self.block_size / 2, y + self.block_size / 2, block_points[:, 2].mean()])
        block_points = block_points - block_center

        # Data augmentation
        if self.augment:
            block_points, block_colors = self._augment(block_points, block_colors)

        # Prepare features
        if self.use_colors:
            features = block_colors
        else:
            features = np.zeros((self.num_points, 0), dtype=np.float32)

        return {
            'points': torch.from_numpy(block_points).float(),
            'features': torch.from_numpy(features).float(),
            'labels': torch.from_numpy(block_labels).long(),
            'block_center': torch.from_numpy(block_center).float()
        }

    def _augment(self, points: np.ndarray, colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply aggressive data augmentation for small datasets"""

        # Random rotation around Z axis (always apply)
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        points = points @ rotation_matrix.T

        # Random rotation around X/Y axis (small angle for realistic augmentation)
        if np.random.random() < 0.5:
            angle = np.random.uniform(-0.1, 0.1)  # ~6 degrees
            axis = np.random.choice(['x', 'y'])
            if axis == 'x':
                rot = np.array([
                    [1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)]
                ])
            else:
                rot = np.array([
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]
                ])
            points = points @ rot.T

        # Random scaling (more aggressive)
        scale = np.random.uniform(0.8, 1.2)
        points = points * scale

        # Anisotropic scaling (different scale per axis)
        if np.random.random() < 0.3:
            scale_xyz = np.random.uniform(0.9, 1.1, size=3)
            points = points * scale_xyz

        # Random translation
        if np.random.random() < 0.5:
            translation = np.random.uniform(-0.2, 0.2, size=3)
            points = points + translation

        # Random jitter (noise)
        points += np.random.normal(0, 0.02, points.shape)

        # Random color jitter (more aggressive)
        if np.random.random() < 0.8:
            # Brightness
            brightness = np.random.uniform(0.8, 1.2)
            colors = colors * brightness

            # Per-channel jitter
            colors = colors + np.random.normal(0, 0.05, colors.shape)

        colors = np.clip(colors, 0, 1)

        # Random point dropout
        if np.random.random() < 0.5:
            drop_ratio = np.random.uniform(0, 0.15)
            num_drop = int(len(points) * drop_ratio)
            if num_drop > 0:
                drop_idx = np.random.choice(len(points), num_drop, replace=False)
                keep_idx = np.random.choice(len(points), num_drop, replace=True)
                points[drop_idx] = points[keep_idx]
                colors[drop_idx] = colors[keep_idx]

        # Random flip along X or Y
        if np.random.random() < 0.5:
            axis = np.random.choice([0, 1])
            points[:, axis] = -points[:, axis]

        return points, colors

    @property
    def num_classes(self) -> int:
        return len(set(self.label_map.values()))

    @property
    def class_names(self) -> List[str]:
        return list(self.label_map.keys())


class ShaftInferenceDataset(Dataset):
    """Dataset for inference on new point clouds"""

    def __init__(
        self,
        las_path: str,
        num_points: int = 4096,
        block_size: float = 1.0,
        stride: float = 0.5,
        use_colors: bool = True
    ):
        self.las_path = Path(las_path)
        self.num_points = num_points
        self.block_size = block_size
        self.stride = stride
        self.use_colors = use_colors

        # Load point cloud
        las = laspy.read(las_path)
        self.points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)

        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            self.colors = np.vstack([las.red, las.green, las.blue]).T.astype(np.float32) / 65535.0
        else:
            self.colors = np.ones((len(self.points), 3), dtype=np.float32) * 0.5

        # Store original coordinates for reconstruction
        self.original_points = self.points.copy()

        # Normalize to origin
        self.centroid = self.points.mean(axis=0)
        self.points = self.points - self.centroid

        # Create blocks
        self.blocks = self._create_blocks()
        print(f"Inference dataset: {len(self.blocks)} blocks from {las_path}")

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

                if len(indices) < 10:
                    continue

                blocks.append((x, y, indices))

        return blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x, y, point_indices = self.blocks[idx]

        block_points = self.points[point_indices].copy()
        block_colors = self.colors[point_indices].copy()

        # Store original indices for reconstruction
        original_indices = point_indices.copy()

        # Sample or pad
        if len(block_points) > self.num_points:
            choice = np.random.choice(len(block_points), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(block_points), self.num_points, replace=True)

        sampled_points = block_points[choice]
        sampled_colors = block_colors[choice]
        sampled_indices = original_indices[choice]

        # Normalize
        block_center = np.array([x + self.block_size / 2, y + self.block_size / 2, sampled_points[:, 2].mean()])
        sampled_points = sampled_points - block_center

        features = sampled_colors if self.use_colors else np.zeros((self.num_points, 0), dtype=np.float32)

        return {
            'points': torch.from_numpy(sampled_points).float(),
            'features': torch.from_numpy(features).float(),
            'indices': torch.from_numpy(sampled_indices).long(),
            'block_center': torch.from_numpy(block_center).float()
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    num_points: int = 4096,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create train and validation dataloaders"""

    train_dataset = ShaftSegmentationDataset(
        data_dir=data_dir,
        num_points=num_points,
        split='train',
        augment=True,
        **kwargs
    )

    val_dataset = ShaftSegmentationDataset(
        data_dir=data_dir,
        num_points=num_points,
        split='val',
        augment=False,
        **kwargs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    dataset_info = {
        'num_classes': train_dataset.num_classes,
        'class_names': train_dataset.class_names,
        'label_map': train_dataset.label_map
    }

    return train_loader, val_loader, dataset_info


if __name__ == '__main__':
    # Test dataset
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '/home/administrator/shaft_segmentation/annotation_tool/annotations'

    train_loader, val_loader, info = create_dataloaders(
        data_dir=data_dir,
        batch_size=4,
        num_points=4096
    )

    print(f"\nDataset info: {info}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Test a batch
    for batch in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Points: {batch['points'].shape}")
        print(f"  Features: {batch['features'].shape}")
        print(f"  Labels: {batch['labels'].shape}")
        print(f"  Label distribution: {torch.bincount(batch['labels'].flatten())}")
        break
