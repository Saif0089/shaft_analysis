#!/usr/bin/env python3
"""
Step 1 v4: Wall Removal using Geometry-Based Detection

Detect wall by geometric properties, not radius:
- WALL: Planar surface with vertical orientation (normal points outward/horizontal)
- PIPES/BEAMS: Linear structures (high linearity)
- PLATFORMS: Planar surfaces with horizontal orientation (normal points up/down)

Uses local PCA to compute geometric descriptors for each point.
"""

import numpy as np
import laspy
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from pathlib import Path


def compute_local_pca_features(points, k=30, verbose=True):
    """
    Compute local PCA features for each point.

    Returns:
        linearity: (λ1 - λ2) / λ1 - high for linear structures
        planarity: (λ2 - λ3) / λ1 - high for planar surfaces
        normals: surface normal (eigenvector of smallest eigenvalue)
    """
    n_points = len(points)

    if verbose:
        print(f"Building KD-tree for {n_points:,} points...")
    tree = KDTree(points)

    linearity = np.zeros(n_points)
    planarity = np.zeros(n_points)
    normals = np.zeros((n_points, 3))

    if verbose:
        print(f"Computing local PCA (k={k})...")

    # Query all neighbors at once
    distances, indices = tree.query(points, k=k+1)

    report_interval = n_points // 10
    for i in range(n_points):
        if verbose and i > 0 and i % report_interval == 0:
            print(f"  {i/n_points*100:.0f}% complete...")

        neighbor_idx = indices[i, 1:]  # Exclude self
        neighbors = points[neighbor_idx]

        # Covariance matrix
        centered = neighbors - np.mean(neighbors, axis=0)
        cov = np.cov(centered.T)

        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort descending
        sort_idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sort_idx]
        eigvecs = eigvecs[:, sort_idx]

        eigvals = np.maximum(eigvals, 1e-10)

        lambda1, lambda2, lambda3 = eigvals

        linearity[i] = (lambda1 - lambda2) / lambda1
        planarity[i] = (lambda2 - lambda3) / lambda1
        normals[i] = eigvecs[:, 2]  # Smallest eigenvalue = surface normal

    if verbose:
        print("Local PCA complete.")

    return linearity, planarity, normals


def classify_geometry(linearity, planarity, normals,
                      linearity_thresh=0.4, planarity_thresh=0.3):
    """
    Classify points by geometric type.

    Returns:
        geometry_type:
            0 = unclassified
            1 = linear (pipes/beams)
            2 = planar-vertical (wall)
            3 = planar-horizontal (platforms)
    """
    n = len(linearity)
    geometry_type = np.zeros(n, dtype=np.int32)

    # Z-component of normal tells us surface orientation
    # |normal_z| ~ 0 means vertical surface (wall)
    # |normal_z| ~ 1 means horizontal surface (platform)
    normal_z = np.abs(normals[:, 2])

    # Linear structures (pipes, beams)
    linear_mask = linearity > linearity_thresh

    # Planar surfaces
    planar_mask = planarity > planarity_thresh

    # Vertical planar = wall (normal is mostly horizontal, so |normal_z| is small)
    vertical_planar = planar_mask & (normal_z < 0.3) & ~linear_mask

    # Horizontal planar = platform (normal is mostly vertical, so |normal_z| is large)
    horizontal_planar = planar_mask & (normal_z > 0.7) & ~linear_mask

    geometry_type[linear_mask] = 1
    geometry_type[vertical_planar] = 2
    geometry_type[horizontal_planar] = 3

    return geometry_type, linear_mask, vertical_planar, horizontal_planar


def plot_geometry_analysis(linearity, planarity, normals, geometry_type, output_path):
    """Plot geometry analysis results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Linearity histogram
    ax = axes[0, 0]
    ax.hist(linearity, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0.4, color='r', linestyle='--', label='Threshold 0.4')
    ax.set_xlabel('Linearity')
    ax.set_ylabel('Count')
    ax.set_title('Linearity Distribution\n(High = pipes/beams)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Planarity histogram
    ax = axes[0, 1]
    ax.hist(planarity, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0.3, color='r', linestyle='--', label='Threshold 0.3')
    ax.set_xlabel('Planarity')
    ax.set_ylabel('Count')
    ax.set_title('Planarity Distribution\n(High = surfaces)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Normal Z-component histogram
    ax = axes[0, 2]
    normal_z = np.abs(normals[:, 2])
    ax.hist(normal_z, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0.3, color='r', linestyle='--', label='Vertical thresh')
    ax.axvline(0.7, color='g', linestyle='--', label='Horizontal thresh')
    ax.set_xlabel('|Normal Z|')
    ax.set_ylabel('Count')
    ax.set_title('Surface Normal Z-Component\n(0=vertical wall, 1=horizontal platform)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Linearity vs Planarity scatter
    ax = axes[1, 0]
    sample_idx = np.random.choice(len(linearity), min(50000, len(linearity)), replace=False)
    colors = ['gray', 'blue', 'red', 'green']
    for gtype in range(4):
        mask = geometry_type[sample_idx] == gtype
        if np.sum(mask) > 0:
            labels = ['Unclassified', 'Linear (pipes)', 'Planar-Vert (wall)', 'Planar-Horiz (platform)']
            ax.scatter(linearity[sample_idx][mask], planarity[sample_idx][mask],
                      s=1, alpha=0.3, c=colors[gtype], label=labels[gtype])
    ax.set_xlabel('Linearity')
    ax.set_ylabel('Planarity')
    ax.set_title('Linearity vs Planarity')
    ax.legend(markerscale=5)
    ax.grid(True, alpha=0.3)

    # 5. Classification summary
    ax = axes[1, 1]
    labels = ['Unclassified', 'Linear\n(pipes/beams)', 'Planar-Vertical\n(WALL)', 'Planar-Horizontal\n(platforms)']
    counts = [np.sum(geometry_type == i) for i in range(4)]
    colors = ['gray', 'blue', 'red', 'green']
    bars = ax.bar(labels, counts, color=colors, edgecolor='black')
    ax.set_ylabel('Point Count')
    ax.set_title('Geometry Classification')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
               f'{count:,}\n({count/len(geometry_type)*100:.1f}%)',
               ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis('off')
    summary = f"""
    Geometry-Based Classification Summary
    =====================================

    Total points: {len(geometry_type):,}

    LINEAR (pipes/beams):     {counts[1]:,} ({counts[1]/len(geometry_type)*100:.1f}%)
       → High linearity, structure elements

    PLANAR-VERTICAL (WALL):   {counts[2]:,} ({counts[2]/len(geometry_type)*100:.1f}%)
       → Planar surface with horizontal normal
       → This is what we want to REMOVE

    PLANAR-HORIZONTAL:        {counts[3]:,} ({counts[3]/len(geometry_type)*100:.1f}%)
       → Platforms, gratings

    UNCLASSIFIED:             {counts[0]:,} ({counts[0]/len(geometry_type)*100:.1f}%)
       → Mixed/transition zones
    """
    ax.text(0.05, 0.5, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_classified_points(points, geometry_type, output_dir):
    """Generate XY and XZ views for each class."""
    labels = ['unclassified', 'linear_pipes', 'wall', 'platforms']
    colors = ['gray', 'blue', 'red', 'green']
    titles = ['Unclassified', 'LINEAR (Pipes/Beams)', 'WALL (to remove)', 'Platforms']

    for gtype in range(4):
        mask = geometry_type == gtype
        if np.sum(mask) == 0:
            continue

        # XY view
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.scatter(points[mask, 0], points[mask, 1], s=0.1, alpha=0.5, c=colors[gtype])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{titles[gtype]}\n({np.sum(mask):,} points, {np.sum(mask)/len(geometry_type)*100:.1f}%)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(output_dir / f"{labels[gtype]}_xy.png"), dpi=150, bbox_inches='tight')
        plt.close()

        # XZ view
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.scatter(points[mask, 0], points[mask, 2], s=0.1, alpha=0.5, c=colors[gtype])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(f'{titles[gtype]} - Side View\n({np.sum(mask):,} points)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(output_dir / f"{labels[gtype]}_xz.png"), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {labels[gtype]}_xy.png, {labels[gtype]}_xz.png")

    # Combined view
    fig, ax = plt.subplots(figsize=(14, 14))
    for gtype in [0, 2, 3, 1]:  # Draw wall first, then pipes on top
        mask = geometry_type == gtype
        if np.sum(mask) > 0:
            ax.scatter(points[mask, 0], points[mask, 1], s=0.1, alpha=0.4,
                      c=colors[gtype], label=f'{titles[gtype]} ({np.sum(mask):,})')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('All Classes Combined')
    ax.set_aspect('equal')
    ax.legend(markerscale=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_dir / "all_classes_xy.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: all_classes_xy.png")


def save_las(filepath, points):
    """Save points to LAS file."""
    new_las = laspy.create(point_format=0, file_version="1.4")
    new_las.x = points[:, 0]
    new_las.y = points[:, 1]
    new_las.z = points[:, 2]
    new_las.write(filepath)
    print(f"Saved: {filepath} ({len(points):,} points)")


def main():
    input_path = "input_data/Navvis5mmShaft14_middle_10m.las"
    output_dir = Path("output/step1_wall_removal_v4_geometry")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1 v4: GEOMETRY-BASED WALL DETECTION")
    print("=" * 60)
    print("\nStrategy:")
    print("  - WALL = planar surface with vertical orientation")
    print("  - PIPES = linear structures (high linearity)")
    print("  - PLATFORMS = planar surface with horizontal orientation")

    # Load data
    print("\nLoading point cloud...")
    las = laspy.read(input_path)
    points = np.vstack((las.x, las.y, las.z)).T
    print(f"Loaded {len(points):,} points")

    # Compute local PCA features
    print("\nComputing local PCA features...")
    linearity, planarity, normals = compute_local_pca_features(points, k=30)

    print(f"\nFeature statistics:")
    print(f"  Linearity: min={linearity.min():.3f}, max={linearity.max():.3f}, mean={linearity.mean():.3f}")
    print(f"  Planarity: min={planarity.min():.3f}, max={planarity.max():.3f}, mean={planarity.mean():.3f}")

    # Classify by geometry
    print("\nClassifying points by geometry...")
    geometry_type, linear_mask, wall_mask, platform_mask = classify_geometry(
        linearity, planarity, normals,
        linearity_thresh=0.4,
        planarity_thresh=0.3
    )

    print(f"\nClassification results:")
    print(f"  Unclassified:        {np.sum(geometry_type==0):,} ({np.sum(geometry_type==0)/len(points)*100:.1f}%)")
    print(f"  Linear (pipes):      {np.sum(geometry_type==1):,} ({np.sum(geometry_type==1)/len(points)*100:.1f}%)")
    print(f"  Planar-Vert (WALL):  {np.sum(geometry_type==2):,} ({np.sum(geometry_type==2)/len(points)*100:.1f}%)")
    print(f"  Planar-Horiz (plat): {np.sum(geometry_type==3):,} ({np.sum(geometry_type==3)/len(points)*100:.1f}%)")

    # Generate visualizations
    print("\n--- Generating visualizations ---")
    plot_geometry_analysis(linearity, planarity, normals, geometry_type,
                          str(output_dir / "00_geometry_analysis.png"))
    plot_classified_points(points, geometry_type, output_dir)

    # Save LAS files
    print("\n--- Saving LAS files ---")

    # Wall = what we want to remove
    save_las(str(output_dir / "wall.las"), points[wall_mask])

    # Interior = everything except wall (pipes + platforms + unclassified)
    interior_mask = ~wall_mask
    save_las(str(output_dir / "interior.las"), points[interior_mask])

    # Also save individual classes
    save_las(str(output_dir / "pipes_linear.las"), points[linear_mask])
    save_las(str(output_dir / "platforms.las"), points[platform_mask])

    # Save parameters
    np.savez(str(output_dir / "geometry_params.npz"),
             linearity=linearity, planarity=planarity, normals=normals,
             geometry_type=geometry_type, wall_mask=wall_mask)
    print(f"Saved: {output_dir / 'geometry_params.npz'}")

    print("\n" + "=" * 60)
    print("STEP 1 v4 COMPLETE")
    print("=" * 60)
    print(f"\nWALL detected: {np.sum(wall_mask):,} points ({np.sum(wall_mask)/len(points)*100:.1f}%)")
    print(f"INTERIOR (to keep): {np.sum(interior_mask):,} points ({np.sum(interior_mask)/len(points)*100:.1f}%)")
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
