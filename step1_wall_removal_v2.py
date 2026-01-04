#!/usr/bin/env python3
"""
Step 1 v2: Wall Removal using Density-Based Outlier Detection

The wall is a dense continuous shell surrounding the interior structure.
We detect it by:
1. Analyzing radial density distribution
2. Finding the dense wall band via histogram analysis
3. Using local density to separate wall from interior
"""

import numpy as np
import laspy
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import KDTree
from pathlib import Path


def compute_radial_from_centroid(points):
    """Compute radial distances from XY centroid."""
    centroid_xy = np.mean(points[:, :2], axis=0)
    radii = np.linalg.norm(points[:, :2] - centroid_xy, axis=1)
    return radii, centroid_xy


def analyze_radial_density(radii, n_bins=100):
    """
    Analyze radial density to find wall band.

    The wall should appear as a high-density peak at outer radii.
    """
    hist, bin_edges = np.histogram(radii, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Compute density (points per unit area in annular ring)
    # Area of annular ring = pi * (r2^2 - r1^2)
    areas = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
    density = hist / areas

    # Smooth for peak detection
    density_smooth = gaussian_filter1d(density, sigma=2)

    return hist, bin_centers, bin_width, density, density_smooth


def find_wall_threshold_from_density(radii, density, bin_centers):
    """
    Find the radius threshold where wall begins.

    Strategy: Find the valley between interior and wall density peaks.
    """
    # Find peaks in density (wall should be a major peak at outer radius)
    peaks, properties = find_peaks(density, height=np.max(density)*0.1, prominence=np.max(density)*0.05)

    if len(peaks) == 0:
        # Fallback: use percentile
        return np.percentile(radii, 70)

    # The wall peak should be at larger radius
    wall_peak_idx = peaks[np.argmax(bin_centers[peaks])]
    wall_peak_radius = bin_centers[wall_peak_idx]

    # Find valley before wall peak (this is where interior ends)
    search_start = max(0, wall_peak_idx - 30)
    valley_region = density[search_start:wall_peak_idx]
    if len(valley_region) > 0:
        valley_idx = search_start + np.argmin(valley_region)
        threshold = bin_centers[valley_idx]
    else:
        threshold = wall_peak_radius - 0.5

    return threshold, wall_peak_radius, peaks, bin_centers[peaks]


def detect_wall_by_local_density(points, radii, k=30):
    """
    Detect wall using local point density.

    Wall points have more uniform, higher local density.
    Interior structure is more sparse/variable.
    """
    print("Computing local density (this may take a moment)...")
    tree = KDTree(points)

    # Get distance to k-th neighbor as density proxy
    distances, _ = tree.query(points, k=k+1)
    kth_distances = distances[:, -1]  # Distance to k-th neighbor

    # Local density ~ 1 / volume of sphere containing k neighbors
    local_density = k / (4/3 * np.pi * kth_distances**3)

    return local_density, kth_distances


def detect_wall_combined(points, radii, local_density):
    """
    Combined approach: wall = outer radius AND high uniform density.
    """
    # Outer points (beyond median radius)
    median_radius = np.median(radii)
    outer_mask = radii > median_radius

    # Among outer points, wall has higher density
    outer_densities = local_density[outer_mask]
    density_threshold = np.percentile(outer_densities, 30)  # Wall is denser

    # Wall = outer AND dense
    high_density_mask = local_density > density_threshold

    # Also check: wall points cluster at specific radii (low radial variance locally)
    return outer_mask & high_density_mask


def plot_radial_analysis(radii, hist, bin_centers, density, density_smooth,
                         threshold, wall_peak_radius, output_path):
    """Plot radial distribution analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Raw histogram
    ax = axes[0, 0]
    ax.bar(bin_centers, hist, width=bin_centers[1]-bin_centers[0], alpha=0.7, edgecolor='black')
    ax.axvline(threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}m')
    ax.set_xlabel('Radial Distance (m)')
    ax.set_ylabel('Point Count')
    ax.set_title('Radial Distribution (Raw Counts)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Density (points per unit area)
    ax = axes[0, 1]
    ax.plot(bin_centers, density, 'b-', alpha=0.5, label='Raw')
    ax.plot(bin_centers, density_smooth, 'b-', linewidth=2, label='Smoothed')
    ax.axvline(threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}m')
    ax.axvline(wall_peak_radius, color='g', linestyle=':', linewidth=2, label=f'Wall peak: {wall_peak_radius:.2f}m')
    ax.set_xlabel('Radial Distance (m)')
    ax.set_ylabel('Density (points/m²)')
    ax.set_title('Radial Density Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Cumulative distribution
    ax = axes[1, 0]
    sorted_radii = np.sort(radii)
    cumulative = np.arange(1, len(radii)+1) / len(radii)
    ax.plot(sorted_radii, cumulative, 'b-', linewidth=2)
    ax.axvline(threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}m')
    ax.axhline(np.sum(radii < threshold)/len(radii), color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('Radial Distance (m)')
    ax.set_ylabel('Cumulative Fraction')
    ax.set_title('Cumulative Radial Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Summary stats
    ax = axes[1, 1]
    ax.axis('off')
    interior_pct = np.sum(radii < threshold) / len(radii) * 100
    wall_pct = 100 - interior_pct
    stats_text = f"""
    Radial Analysis Summary
    ========================

    Total points: {len(radii):,}

    Radial range: {radii.min():.2f}m - {radii.max():.2f}m
    Median radius: {np.median(radii):.2f}m

    Detected threshold: {threshold:.2f}m
    Wall density peak: {wall_peak_radius:.2f}m

    Interior (r < {threshold:.2f}m): {np.sum(radii < threshold):,} ({interior_pct:.1f}%)
    Wall (r >= {threshold:.2f}m): {np.sum(radii >= threshold):,} ({wall_pct:.1f}%)
    """
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_xy_view(points, mask, title, filename, centroid=None, threshold_radius=None):
    """Generate top-down XY view."""
    fig, ax = plt.subplots(figsize=(12, 12))

    selected = points[mask]
    ax.scatter(selected[:, 0], selected[:, 1], s=0.1, alpha=0.5, c='steelblue')

    if centroid is not None and threshold_radius is not None:
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = centroid[0] + threshold_radius * np.cos(theta)
        circle_y = centroid[1] + threshold_radius * np.sin(theta)
        ax.plot(circle_x, circle_y, 'r--', linewidth=2, label=f'Threshold r={threshold_radius:.2f}m')
        ax.plot(centroid[0], centroid[1], 'r+', markersize=15, markeredgewidth=2)
        ax.legend()

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{title}\n({np.sum(mask):,} points)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_xz_view(points, mask, title, filename):
    """Generate side XZ view."""
    fig, ax = plt.subplots(figsize=(14, 6))

    selected = points[mask]
    ax.scatter(selected[:, 0], selected[:, 2], s=0.1, alpha=0.5, c='steelblue')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title(f'{title}\n({np.sum(mask):,} points)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


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
    output_dir = Path("output/step1_wall_removal_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1 v2: WALL REMOVAL - DENSITY-BASED APPROACH")
    print("=" * 60)

    # Load data
    print("\nLoading point cloud...")
    las = laspy.read(input_path)
    points = np.vstack((las.x, las.y, las.z)).T
    print(f"Loaded {len(points):,} points")

    # Compute radial distances
    print("\nComputing radial distances from centroid...")
    radii, centroid = compute_radial_from_centroid(points)
    print(f"Centroid: ({centroid[0]:.3f}, {centroid[1]:.3f})")
    print(f"Radial range: {radii.min():.3f}m - {radii.max():.3f}m")

    # Analyze radial density
    print("\nAnalyzing radial density distribution...")
    hist, bin_centers, bin_width, density, density_smooth = analyze_radial_density(radii, n_bins=80)

    # Find wall threshold from density analysis
    result = find_wall_threshold_from_density(radii, density_smooth, bin_centers)
    threshold, wall_peak_radius, peaks, peak_radii = result

    print(f"\nDensity analysis results:")
    print(f"  Detected density peaks at radii: {peak_radii}")
    print(f"  Wall peak radius: {wall_peak_radius:.3f}m")
    print(f"  Suggested threshold: {threshold:.3f}m")

    # Test multiple thresholds
    print("\n--- Testing different thresholds ---")
    test_thresholds = [threshold - 0.3, threshold - 0.15, threshold, threshold + 0.15, threshold + 0.3]
    for t in test_thresholds:
        n_interior = np.sum(radii < t)
        n_wall = np.sum(radii >= t)
        marker = " <-- auto-detected" if abs(t - threshold) < 0.01 else ""
        print(f"  r < {t:.2f}m: Interior={n_interior:,} ({n_interior/len(radii)*100:.1f}%), "
              f"Wall={n_wall:,} ({n_wall/len(radii)*100:.1f}%){marker}")

    # Apply threshold
    interior_mask = radii < threshold
    wall_mask = ~interior_mask

    print(f"\n--- Using threshold = {threshold:.2f}m ---")
    print(f"Interior: {np.sum(interior_mask):,} points ({np.sum(interior_mask)/len(points)*100:.1f}%)")
    print(f"Wall: {np.sum(wall_mask):,} points ({np.sum(wall_mask)/len(points)*100:.1f}%)")

    # Generate visualizations
    print("\n--- Generating visualizations ---")

    # Radial analysis plot
    plot_radial_analysis(radii, hist, bin_centers, density, density_smooth,
                        threshold, wall_peak_radius,
                        str(output_dir / "01_radial_analysis.png"))

    # Wall points (top and side)
    plot_xy_view(points, wall_mask, "WALL Points",
                str(output_dir / "02_wall_xy.png"), centroid, threshold)
    plot_xz_view(points, wall_mask, "WALL Points - Side View",
                str(output_dir / "03_wall_xz.png"))

    # Interior points (top and side)
    plot_xy_view(points, interior_mask, "INTERIOR Points",
                str(output_dir / "04_interior_xy.png"), centroid, threshold)
    plot_xz_view(points, interior_mask, "INTERIOR Points - Side View",
                str(output_dir / "05_interior_xz.png"))

    # Combined view showing both
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(points[wall_mask, 0], points[wall_mask, 1], s=0.1, alpha=0.3, c='gray', label='Wall')
    ax.scatter(points[interior_mask, 0], points[interior_mask, 1], s=0.1, alpha=0.5, c='red', label='Interior')
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = centroid[0] + threshold * np.cos(theta)
    circle_y = centroid[1] + threshold * np.sin(theta)
    ax.plot(circle_x, circle_y, 'g--', linewidth=2, label=f'Threshold r={threshold:.2f}m')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Wall vs Interior Separation\nThreshold: {threshold:.2f}m')
    ax.set_aspect('equal')
    ax.legend(markerscale=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_dir / "06_combined_xy.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / '06_combined_xy.png'}")

    # Save LAS files
    print("\n--- Saving LAS files ---")
    save_las(str(output_dir / "wall.las"), points[wall_mask])
    save_las(str(output_dir / "interior.las"), points[interior_mask])

    # Save parameters
    np.savez(str(output_dir / "wall_params.npz"),
             centroid=centroid, threshold=threshold,
             wall_peak_radius=wall_peak_radius,
             wall_mask=wall_mask, interior_mask=interior_mask)
    print(f"Saved: {output_dir / 'wall_params.npz'}")

    print("\n" + "=" * 60)
    print("STEP 1 v2 COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nIf threshold needs adjustment, modify and re-run.")


if __name__ == "__main__":
    main()
