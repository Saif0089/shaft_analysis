#!/usr/bin/env python3
"""
Step 2: Separate Curved Pipes from Straight Steel

Input: output/step1_wall_stripped/interior.las
Output: output/step2_pipes_extracted/
    - curved_pipes.las (arched/curved vertical elements)
    - straight_steel.las (rectangular grid structure)

Algorithm:
1. Cluster points into vertical elements using DBSCAN in XY plane
2. For each element, compute center displacement (XY drift from bottom to top)
3. Classify: displacement > 0.15m -> CURVED

Key Insight:
The distinction between curved pipes and straight steel is MACRO-LEVEL curvature,
not local geometric features. Locally, a segment of curved pipe looks identical
to a straight beam. The difference is how much the XY position drifts along Z.
"""

import numpy as np
import laspy
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================
DBSCAN_EPS = 0.15           # Spatial tolerance for clustering (meters)
DBSCAN_MIN_SAMPLES = 50     # Minimum points per cluster

DISPLACEMENT_THRESHOLD = 0.15   # Center displacement threshold (meters)
# Note: Width alone is unreliable - connected straight beams have large XY span
# Use displacement as primary indicator of actual curvature


def load_las(filepath):
    """Load LAS file and return points."""
    las = laspy.read(filepath)
    points = np.vstack((las.x, las.y, las.z)).T
    print(f"Loaded {len(points):,} points from {filepath}")
    return points


def save_las(filepath, points):
    """Save points to LAS file."""
    if len(points) == 0:
        print(f"Skipping {filepath} - no points")
        return
    new_las = laspy.create(point_format=0, file_version="1.4")
    new_las.x = points[:, 0]
    new_las.y = points[:, 1]
    new_las.z = points[:, 2]
    new_las.write(filepath)
    print(f"Saved: {filepath} ({len(points):,} points)")


def cluster_vertical_elements(points):
    """
    Cluster points into vertical elements using DBSCAN in XY plane.

    Returns: labels array (-1 = noise)
    """
    print(f"\nClustering points in XY plane (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})...")

    xy_points = points[:, :2]  # Only X, Y
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(xy_points)
    labels = clustering.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    print(f"  Found {n_clusters} clusters, {n_noise:,} noise points")

    return labels


def compute_element_features(points, labels):
    """
    Compute macro-level features for each vertical element.

    Returns dict with:
        - cluster_id: cluster label
        - n_points: number of points
        - center_displacement: XY drift from bottom to top
        - xy_width: cross-sectional spread
        - z_range: height extent
    """
    unique_labels = [l for l in np.unique(labels) if l >= 0]
    features = []

    print(f"\nComputing features for {len(unique_labels)} elements...")

    for label in unique_labels:
        mask = labels == label
        element_points = points[mask]
        n_points = len(element_points)

        # Z range
        z_vals = element_points[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        z_range = z_max - z_min

        # Skip very short elements
        if z_range < 1.0:
            continue

        # Center displacement: XY drift from bottom to top
        bottom_threshold = z_min + 0.2 * z_range
        top_threshold = z_max - 0.2 * z_range

        bottom_mask = z_vals < bottom_threshold
        top_mask = z_vals > top_threshold

        if np.sum(bottom_mask) < 10 or np.sum(top_mask) < 10:
            # Not enough points at top/bottom
            displacement = 0.0
        else:
            bottom_center = element_points[bottom_mask, :2].mean(axis=0)
            top_center = element_points[top_mask, :2].mean(axis=0)
            displacement = np.linalg.norm(top_center - bottom_center)

        # XY width: cross-sectional spread
        xy_std = element_points[:, :2].std(axis=0)
        xy_width = np.sqrt(xy_std[0]**2 + xy_std[1]**2)

        features.append({
            'cluster_id': label,
            'n_points': n_points,
            'center_displacement': displacement,
            'xy_width': xy_width,
            'z_range': z_range
        })

    print(f"  Computed features for {len(features)} valid elements")

    return features


def classify_elements(features):
    """
    Classify elements as curved or straight based on center displacement.

    Rule: displacement > 0.15m -> CURVED
    (Width alone is unreliable - connected horizontal beams have large XY span)
    """
    print(f"\nClassifying elements...")
    print(f"  Threshold: center displacement > {DISPLACEMENT_THRESHOLD}m -> CURVED")

    curved_ids = []
    straight_ids = []

    for f in features:
        is_curved = f['center_displacement'] > DISPLACEMENT_THRESHOLD

        if is_curved:
            curved_ids.append(f['cluster_id'])
        else:
            straight_ids.append(f['cluster_id'])

        status = "CURVED" if is_curved else "STRAIGHT"
        print(f"    Cluster {f['cluster_id']:2d}: {f['n_points']:6,} pts, "
              f"disp={f['center_displacement']:.3f}m, width={f['xy_width']:.3f}m -> {status}")

    print(f"\n  CURVED elements: {len(curved_ids)}")
    print(f"  STRAIGHT elements: {len(straight_ids)}")

    return curved_ids, straight_ids


def create_masks(labels, curved_ids, straight_ids):
    """Create boolean masks for curved and straight points."""
    curved_mask = np.isin(labels, curved_ids)
    straight_mask = np.isin(labels, straight_ids)

    # Include noise points in straight (they're likely small structural elements)
    noise_mask = labels == -1
    straight_mask = straight_mask | noise_mask

    return curved_mask, straight_mask


def plot_results(points, curved_mask, straight_mask, features, output_dir):
    """Generate visualization of results."""

    # Top view with classification
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: Top view
    ax = axes[0]
    if np.sum(straight_mask) > 0:
        ax.scatter(points[straight_mask, 0], points[straight_mask, 1],
                   s=0.1, alpha=0.3, c='blue', label=f'Straight Steel ({np.sum(straight_mask):,})')
    if np.sum(curved_mask) > 0:
        ax.scatter(points[curved_mask, 0], points[curved_mask, 1],
                   s=0.1, alpha=0.5, c='red', label=f'Curved Pipes ({np.sum(curved_mask):,})')
    ax.set_title('Step 2: Curved Pipes vs Straight Steel - Top View (XY)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.legend(markerscale=10)
    ax.grid(True, alpha=0.3)

    # Right: Feature space
    ax = axes[1]
    for f in features:
        is_curved = f['center_displacement'] > DISPLACEMENT_THRESHOLD
        color = 'red' if is_curved else 'blue'
        marker = '^' if is_curved else 's'
        ax.scatter(f['xy_width'], f['center_displacement'],
                   s=100, c=color, marker=marker, edgecolors='black', linewidths=0.5)

    # Decision boundary (displacement only)
    ax.axhline(y=DISPLACEMENT_THRESHOLD, color='gray', linestyle='--', alpha=0.7,
               label=f'Displacement threshold ({DISPLACEMENT_THRESHOLD}m)')

    ax.set_xlabel('XY Width (m)')
    ax.set_ylabel('Center Displacement (m)')
    ax.set_title('Feature Space: Classification Thresholds')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_dir / "step2_results.png"), dpi=150)
    plt.close()
    print(f"Saved: step2_results.png")

    # Side view
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # XZ view
    ax = axes[0]
    if np.sum(straight_mask) > 0:
        ax.scatter(points[straight_mask, 0], points[straight_mask, 2],
                   s=0.1, alpha=0.3, c='blue', label='Straight Steel')
    if np.sum(curved_mask) > 0:
        ax.scatter(points[curved_mask, 0], points[curved_mask, 2],
                   s=0.1, alpha=0.5, c='red', label='Curved Pipes')
    ax.set_title('Side View (XZ)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.legend(markerscale=10)
    ax.grid(True, alpha=0.3)

    # YZ view
    ax = axes[1]
    if np.sum(straight_mask) > 0:
        ax.scatter(points[straight_mask, 1], points[straight_mask, 2],
                   s=0.1, alpha=0.3, c='blue', label='Straight Steel')
    if np.sum(curved_mask) > 0:
        ax.scatter(points[curved_mask, 1], points[curved_mask, 2],
                   s=0.1, alpha=0.5, c='red', label='Curved Pipes')
    ax.set_title('Side View (YZ)')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('Z (m)')
    ax.legend(markerscale=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_dir / "step2_sideview.png"), dpi=150)
    plt.close()
    print(f"Saved: step2_sideview.png")


def main():
    # =========================================================================
    # Configuration
    # =========================================================================
    input_path = Path("output/step1_wall_stripped/interior.las")
    output_dir = Path("output/step2_pipes_extracted")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 2: SEPARATE CURVED PIPES FROM STRAIGHT STEEL")
    print("=" * 70)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_dir}")
    print("\nAlgorithm:")
    print("  1. Cluster points into vertical elements (DBSCAN in XY)")
    print("  2. Compute center displacement (XY drift from bottom to top)")
    print("  3. Classify: displacement > 0.15m -> CURVED")

    # =========================================================================
    # Step 1: Load interior points
    # =========================================================================
    print("\n" + "-" * 70)
    print("[1/5] Loading interior points...")
    print("-" * 70)
    points = load_las(str(input_path))

    # =========================================================================
    # Step 2: Cluster into vertical elements
    # =========================================================================
    print("\n" + "-" * 70)
    print("[2/5] Clustering into vertical elements...")
    print("-" * 70)
    labels = cluster_vertical_elements(points)

    # =========================================================================
    # Step 3: Compute element features
    # =========================================================================
    print("\n" + "-" * 70)
    print("[3/5] Computing element features...")
    print("-" * 70)
    features = compute_element_features(points, labels)

    # =========================================================================
    # Step 4: Classify elements
    # =========================================================================
    print("\n" + "-" * 70)
    print("[4/5] Classifying elements...")
    print("-" * 70)
    curved_ids, straight_ids = classify_elements(features)
    curved_mask, straight_mask = create_masks(labels, curved_ids, straight_ids)

    # =========================================================================
    # Step 5: Save results
    # =========================================================================
    print("\n" + "-" * 70)
    print("[5/5] Saving results...")
    print("-" * 70)

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL CLASSIFICATION")
    print(f"{'='*70}")
    print(f"CURVED PIPES:    {np.sum(curved_mask):,} points ({np.sum(curved_mask)/len(points)*100:.1f}%)")
    print(f"STRAIGHT STEEL:  {np.sum(straight_mask):,} points ({np.sum(straight_mask)/len(points)*100:.1f}%)")

    # Generate visualizations
    print("\n--- Generating visualizations ---")
    plot_results(points, curved_mask, straight_mask, features, output_dir)

    # Save LAS files
    print("\n--- Saving LAS files ---")
    save_las(str(output_dir / "curved_pipes.las"), points[curved_mask])
    save_las(str(output_dir / "straight_steel.las"), points[straight_mask])

    print(f"\n{'='*70}")
    print("STEP 2 COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  - curved_pipes.las   : Arched/curved elements")
    print(f"  - straight_steel.las : Rectangular grid structure")
    print(f"\nNext step: straight_steel.las ready for beam/bunton classification")


if __name__ == "__main__":
    main()
