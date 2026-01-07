"""Dataset for point cloud semantic segmentation."""
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.las_io import read_las


class ShaftPointCloudDataset(Dataset):
    """Dataset for shaft point cloud segmentation."""

    def __init__(self, las_files: List[str], num_points: int = 8192,
                 augment: bool = True, normalize: bool = True):
        """
        Args:
            las_files: List of LAS file paths with ground truth labels
            num_points: Number of points to sample per item
            augment: Whether to apply data augmentation
            normalize: Whether to normalize coordinates
        """
        self.las_files = las_files
        self.num_points = num_points
        self.augment = augment
        self.normalize = normalize

        # Load all data into memory
        self.all_points = []
        self.all_labels = []

        for filepath in las_files:
            data = read_las(filepath)
            xyz = data['xyz']
            labels = data['classification']

            if labels is None:
                raise ValueError(f"No classification labels in {filepath}")

            self.all_points.append(xyz)
            self.all_labels.append(labels)

        # Concatenate all data
        self.points = np.concatenate(self.all_points, axis=0)
        self.labels = np.concatenate(self.all_labels, axis=0)

        # Calculate global bounds for normalization
        self.center = self.points.mean(axis=0)
        self.scale = np.abs(self.points - self.center).max()

        print(f"Loaded {len(self.points):,} points from {len(las_files)} files")
        print(f"Unique labels: {np.unique(self.labels)}")

    def __len__(self) -> int:
        # Number of samples we can extract
        return max(1, len(self.points) // self.num_points)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Random sample from all points
        if len(self.points) <= self.num_points:
            indices = np.random.choice(len(self.points), self.num_points, replace=True)
        else:
            indices = np.random.choice(len(self.points), self.num_points, replace=False)

        points = self.points[indices].copy()
        labels = self.labels[indices].copy()

        # Normalize
        if self.normalize:
            points = (points - self.center) / self.scale

        # Augmentation
        if self.augment:
            points = self._augment(points)

        # Convert to tensors
        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels).long()

        return points, labels

    def _augment(self, points: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        # Random rotation around Z axis
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t, 0],
            [0, 0, 1]
        ])
        points = points @ rotation_matrix.T

        # Random jitter
        points += np.random.normal(0, 0.01, points.shape)

        # Random scaling
        scale = np.random.uniform(0.9, 1.1)
        points *= scale

        return points

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced data."""
        unique, counts = np.unique(self.labels, return_counts=True)
        total = counts.sum()

        # Inverse frequency weighting
        weights = total / (len(unique) * counts)

        # Create full weight array (including missing classes)
        max_class = unique.max() + 1
        full_weights = np.ones(max_class)
        for cls, weight in zip(unique, weights):
            full_weights[cls] = weight

        return torch.from_numpy(full_weights).float()


class InferenceDataset(Dataset):
    """Dataset for inference (no labels required)."""

    def __init__(self, las_file: str, num_points: int = 8192,
                 overlap: float = 0.5):
        """
        Args:
            las_file: LAS file path
            num_points: Number of points per batch
            overlap: Overlap ratio between batches
        """
        self.num_points = num_points

        # Load data
        data = read_las(las_file)
        self.points = data['xyz']
        self.original_indices = np.arange(len(self.points))

        # Calculate global bounds for normalization
        self.center = self.points.mean(axis=0)
        self.scale = np.abs(self.points - self.center).max()

        # Create batches with overlap
        stride = int(num_points * (1 - overlap))
        self.batch_indices = []

        for start in range(0, len(self.points), stride):
            end = min(start + num_points, len(self.points))
            indices = np.arange(start, end)

            # Pad if necessary
            if len(indices) < num_points:
                pad_indices = np.random.choice(len(self.points), num_points - len(indices))
                indices = np.concatenate([indices, pad_indices])

            self.batch_indices.append(indices)

        print(f"Created {len(self.batch_indices)} inference batches")

    def __len__(self) -> int:
        return len(self.batch_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        indices = self.batch_indices[idx]
        points = self.points[indices].copy()

        # Normalize
        points = (points - self.center) / self.scale

        # Convert to tensor
        points = torch.from_numpy(points).float()

        return points, indices


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import GROUND_TRUTH_DIR, SLICES_DIR

    # Test with existing ground truth if available
    gt_files = [f for f in os.listdir(GROUND_TRUTH_DIR) if f.endswith('_gt.las')]

    if gt_files:
        gt_path = os.path.join(GROUND_TRUTH_DIR, gt_files[0])
        print(f"Testing with {gt_path}")

        dataset = ShaftPointCloudDataset([gt_path], num_points=4096)
        print(f"Dataset length: {len(dataset)}")

        points, labels = dataset[0]
        print(f"Points shape: {points.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels unique: {torch.unique(labels)}")

        weights = dataset.get_class_weights()
        print(f"Class weights: {weights}")
    else:
        print("No ground truth files found. Create some annotations first.")

    # Test inference dataset
    slice_files = [f for f in os.listdir(SLICES_DIR) if f.endswith('.las')]
    if slice_files:
        slice_path = os.path.join(SLICES_DIR, slice_files[0])
        print(f"\nTesting inference dataset with {slice_path}")

        inf_dataset = InferenceDataset(slice_path, num_points=4096)
        print(f"Dataset length: {len(inf_dataset)}")

        points, indices = inf_dataset[0]
        print(f"Points shape: {points.shape}")
        print(f"Indices shape: {indices.shape}")

    print("\nDataset tests passed!")
