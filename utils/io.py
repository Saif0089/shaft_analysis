"""
I/O utilities for loading and saving point cloud data.
"""

import numpy as np
import laspy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LABEL_COLORS, LABEL_NAMES


def load_las(filepath: str) -> tuple[np.ndarray, laspy.LasData]:
    """
    Load a LAS file and return points as numpy array.

    Args:
        filepath: Path to LAS file

    Returns:
        points: Nx3 array of XYZ coordinates
        las: Original LasData object for metadata preservation
    """
    las = laspy.read(filepath)
    points = np.vstack((las.x, las.y, las.z)).T
    print(f"Loaded {len(points):,} points from {filepath}")
    return points, las


def save_las(filepath: str, points: np.ndarray, source_las: laspy.LasData = None):
    """
    Save points to a LAS file.

    Args:
        filepath: Output path
        points: Nx3 array of XYZ coordinates
        source_las: Optional source LAS for header/format copying
    """
    if source_las is not None:
        # Create new LAS with same format
        new_las = laspy.create(point_format=source_las.header.point_format,
                               file_version=source_las.header.version)
    else:
        new_las = laspy.create(point_format=0, file_version="1.4")

    new_las.x = points[:, 0]
    new_las.y = points[:, 1]
    new_las.z = points[:, 2]

    new_las.write(filepath)
    print(f"Saved {len(points):,} points to {filepath}")


def save_colored_las(filepath: str, points: np.ndarray, labels: np.ndarray,
                     source_las: laspy.LasData = None):
    """
    Save points with RGB colors based on labels.

    Args:
        filepath: Output path
        points: Nx3 array of XYZ coordinates
        labels: N array of integer labels
        source_las: Optional source LAS for header copying
    """
    # Create LAS with RGB support (format 2 or 3)
    new_las = laspy.create(point_format=2, file_version="1.4")

    new_las.x = points[:, 0]
    new_las.y = points[:, 1]
    new_las.z = points[:, 2]

    # Assign colors based on labels
    colors = np.zeros((len(points), 3), dtype=np.uint16)
    for label, rgb in LABEL_COLORS.items():
        mask = labels == label
        # LAS uses 16-bit colors (0-65535)
        colors[mask] = [c * 256 for c in rgb]

    new_las.red = colors[:, 0]
    new_las.green = colors[:, 1]
    new_las.blue = colors[:, 2]

    new_las.write(filepath)
    print(f"Saved colored LAS ({len(points):,} points) to {filepath}")


def save_separate_las(output_dir: str, points: np.ndarray, labels: np.ndarray,
                      source_las: laspy.LasData = None):
    """
    Save separate LAS files for each label class.

    Args:
        output_dir: Output directory
        points: Nx3 array of XYZ coordinates
        labels: N array of integer labels
        source_las: Optional source LAS for header copying
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        label_points = points[mask]

        if len(label_points) == 0:
            continue

        name = LABEL_NAMES.get(label, f"class_{label}")
        filepath = output_path / f"{name}.las"
        save_las(str(filepath), label_points, source_las)


def save_numpy(output_dir: str, points: np.ndarray, labels: np.ndarray,
               features: dict = None):
    """
    Save points, labels, and optional features as numpy arrays.

    Args:
        output_dir: Output directory
        points: Nx3 array of XYZ coordinates
        labels: N array of integer labels
        features: Optional dict of feature arrays
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / "points.npy", points)
    np.save(output_path / "labels.npy", labels)
    print(f"Saved numpy arrays to {output_path}")

    if features:
        for name, arr in features.items():
            np.save(output_path / f"{name}.npy", arr)
            print(f"  Saved {name}.npy")


def print_label_stats(labels: np.ndarray):
    """Print statistics about label distribution."""
    print("\n=== Label Statistics ===")
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    for label, count in zip(unique, counts):
        name = LABEL_NAMES.get(label, f"class_{label}")
        pct = count / total * 100
        print(f"  {name}: {count:,} points ({pct:.1f}%)")
    print()
