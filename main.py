#!/usr/bin/env python3
"""
Mine Shaft Steel Structure Segmentation Pipeline

This pipeline segments a point cloud of a mine shaft into:
- Wall (cylindrical shaft surface)
- Vertical members (columns, vertical beams)
- Horizontal members (beams, pipes) with orientation
- Platforms/gratings

Usage:
    python main.py <input.las> [output_dir]
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import time

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (LABEL_WALL, LABEL_VERTICAL, LABEL_HORIZONTAL_0,
                    LABEL_HORIZONTAL_90, LABEL_PLATFORM, LABEL_UNCLASSIFIED)
from utils.io import (load_las, save_colored_las, save_separate_las,
                      save_numpy, print_label_stats)
from preprocess.wall_removal import remove_wall
from features.local_pca import compute_local_pca
from detectors.vertical_members import detect_vertical_members
from detectors.horizontal_members import detect_horizontal_members
from detectors.platforms import detect_platforms
from postprocess.split_intersections import split_intersections


def segment_shaft(input_path: str, output_dir: str = None) -> tuple:
    """
    Main segmentation pipeline.

    Args:
        input_path: Path to input LAS file
        output_dir: Output directory (default: ./output)

    Returns:
        points: Nx3 array of all points
        labels: N array of final labels
        features: Dictionary of computed features
    """
    start_time = time.time()

    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MINE SHAFT STEEL STRUCTURE SEGMENTATION")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print()

    # =========================================================================
    # Step 1: Load data
    # =========================================================================
    print("[1/6] Loading point cloud...")
    points, source_las = load_las(input_path)
    n_points = len(points)

    # Initialize labels (all unclassified)
    labels = np.full(n_points, LABEL_UNCLASSIFIED, dtype=np.int32)

    # =========================================================================
    # Step 2: Remove wall
    # =========================================================================
    print("\n[2/6] Removing wall...")
    wall_mask, interior_mask, center_xy, radii = remove_wall(points)
    labels[wall_mask] = LABEL_WALL

    # Get interior points for further processing
    interior_indices = np.where(interior_mask)[0]
    interior_points = points[interior_indices]

    print(f"  Processing {len(interior_points):,} interior points")

    # =========================================================================
    # Step 3: Compute local PCA features
    # =========================================================================
    print("\n[3/6] Computing local PCA features...")
    features = compute_local_pca(interior_points, verbose=True)

    # =========================================================================
    # Step 4: Detect vertical members
    # =========================================================================
    print("\n[4/6] Detecting vertical members...")
    vertical_mask, vertical_clusters = detect_vertical_members(
        interior_points, features
    )

    # Map back to full point cloud
    for i, idx in enumerate(interior_indices):
        if vertical_mask[i]:
            labels[idx] = LABEL_VERTICAL

    already_classified = vertical_mask.copy()

    # =========================================================================
    # Step 5: Detect horizontal members
    # =========================================================================
    print("\n[5/6] Detecting horizontal members...")
    horizontal_mask, orientation_labels, horizontal_clusters = detect_horizontal_members(
        interior_points, features, already_classified
    )

    # Map back to full point cloud with orientation
    for i, idx in enumerate(interior_indices):
        if horizontal_mask[i]:
            labels[idx] = orientation_labels[i]

    already_classified = already_classified | horizontal_mask

    # Post-process: split at intersections
    if np.any(horizontal_mask):
        print("  Post-processing: splitting intersections...")
        horizontal_clusters = split_intersections(
            interior_points, horizontal_clusters, features
        )

    # =========================================================================
    # Step 6: Detect platforms
    # =========================================================================
    print("\n[6/6] Detecting platforms...")
    platform_mask, platform_clusters, z_levels = detect_platforms(
        interior_points, features, already_classified
    )

    # Map back to full point cloud
    for i, idx in enumerate(interior_indices):
        if platform_mask[i]:
            labels[idx] = LABEL_PLATFORM

    # =========================================================================
    # Results summary
    # =========================================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("SEGMENTATION COMPLETE")
    print("=" * 60)
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print_label_stats(labels)

    # =========================================================================
    # Save outputs
    # =========================================================================
    print("Saving outputs...")

    # Colored LAS
    colored_path = output_dir / "segmented_colored.las"
    save_colored_las(str(colored_path), points, labels, source_las)

    # Separate LAS files
    save_separate_las(str(output_dir), points, labels, source_las)

    # NumPy arrays
    # Map features back to full point cloud for saving
    full_features = {}
    for key in ['linearity', 'planarity', 'scattering']:
        full_arr = np.zeros(n_points)
        full_arr[interior_indices] = features[key]
        full_features[key] = full_arr

    save_numpy(str(output_dir), points, labels, full_features)

    print(f"\nOutputs saved to: {output_dir}")
    print("  - segmented_colored.las (single colored file)")
    print("  - wall.las, vertical.las, horizontal_*.las, platforms.las, unclassified.las")
    print("  - points.npy, labels.npy, linearity.npy, planarity.npy, scattering.npy")

    return points, labels, features


def main():
    parser = argparse.ArgumentParser(
        description="Segment mine shaft point cloud into steel structure components"
    )
    parser.add_argument(
        "input",
        help="Input LAS file path"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory (default: ./output)"
    )

    args = parser.parse_args()

    segment_shaft(args.input, args.output)


if __name__ == "__main__":
    main()
