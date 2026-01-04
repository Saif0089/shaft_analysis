#!/usr/bin/env python3
"""
Step 1 v6: Wall Removal using Cylindrical Unwrap + Radial Envelope

The shaft wall is:
- Global, continuous, dominant in angular coverage
- Radially outermost structure
- Relatively smooth

Strategy:
1. Estimate shaft axis (PCA)
2. Transform to cylindrical coordinates (r, θ, z)
3. Bin by (θ, z) and find radial envelope (upper percentile)
4. Wall = points near the envelope

This handles pipes attached to wall (they stick inward → smaller r → kept)
"""

import numpy as np
import laspy
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter
from pathlib import Path


def estimate_shaft_axis(points):
    """
    Estimate shaft axis using PCA.
    First principal component ≈ shaft axis (Z direction).
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # Covariance and PCA
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort descending
    sort_idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, sort_idx]

    # First eigenvector = shaft axis (should be mostly Z)
    axis = eigvecs[:, 0]

    # Make sure it points "up" (positive Z)
    if axis[2] < 0:
        axis = -axis

    return centroid, axis


def to_cylindrical(points, center, axis):
    """
    Transform points to cylindrical coordinates relative to shaft axis.

    Returns:
        r: radial distance from axis
        theta: angle around axis (0 to 2π)
        z: distance along axis
    """
    # Translate to center
    translated = points - center

    # Project onto axis to get z
    z = np.dot(translated, axis)

    # Get radial component (perpendicular to axis)
    radial_vec = translated - np.outer(z, axis)

    # Compute r (distance from axis)
    r = np.linalg.norm(radial_vec, axis=1)

    # Compute theta (angle around axis)
    # Create orthonormal basis perpendicular to axis
    # Find a vector not parallel to axis
    if abs(axis[0]) < 0.9:
        perp1 = np.cross(axis, [1, 0, 0])
    else:
        perp1 = np.cross(axis, [0, 1, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(axis, perp1)

    # Project radial vector onto perp1, perp2 to get x', y'
    x_prime = np.dot(radial_vec, perp1)
    y_prime = np.dot(radial_vec, perp2)

    theta = np.arctan2(y_prime, x_prime)  # -π to π
    theta = (theta + 2*np.pi) % (2*np.pi)  # 0 to 2π

    return r, theta, z


def compute_radial_envelope(r, theta, z, theta_bins=180, z_bins=50, percentile=95):
    """
    Compute radial envelope r_wall(θ, z) by binning and taking upper percentile.

    Returns:
        envelope: 2D array of r_wall values
        theta_edges, z_edges: bin edges
        theta_centers, z_centers: bin centers
    """
    theta_edges = np.linspace(0, 2*np.pi, theta_bins + 1)
    z_edges = np.linspace(z.min(), z.max(), z_bins + 1)

    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2

    # Initialize envelope with NaN
    envelope = np.full((theta_bins, z_bins), np.nan)

    # Digitize points into bins
    theta_idx = np.digitize(theta, theta_edges) - 1
    z_idx = np.digitize(z, z_edges) - 1

    # Clamp to valid range
    theta_idx = np.clip(theta_idx, 0, theta_bins - 1)
    z_idx = np.clip(z_idx, 0, z_bins - 1)

    # For each bin, compute upper percentile of r
    for ti in range(theta_bins):
        for zi in range(z_bins):
            mask = (theta_idx == ti) & (z_idx == zi)
            if np.sum(mask) > 5:  # Need enough points
                envelope[ti, zi] = np.percentile(r[mask], percentile)

    # Interpolate missing values (gaps in scan)
    # Use median filter to fill and smooth
    envelope_filled = envelope.copy()

    # Fill NaN with neighbor median
    for _ in range(3):  # Iterate a few times
        nan_mask = np.isnan(envelope_filled)
        if not np.any(nan_mask):
            break
        envelope_smooth = median_filter(np.nan_to_num(envelope_filled, nan=np.nanmedian(envelope_filled)), size=3)
        envelope_filled[nan_mask] = envelope_smooth[nan_mask]

    # Smooth the envelope
    envelope_smooth = gaussian_filter(envelope_filled, sigma=1)

    return envelope_smooth, theta_edges, z_edges, theta_centers, z_centers


def classify_wall(r, theta, z, envelope, theta_edges, z_edges, epsilon=0.1):
    """
    Classify points as wall if |r - r_wall(θ, z)| < epsilon
    """
    theta_bins = len(theta_edges) - 1
    z_bins = len(z_edges) - 1

    # Get bin indices for each point
    theta_idx = np.digitize(theta, theta_edges) - 1
    z_idx = np.digitize(z, z_edges) - 1

    # Clamp
    theta_idx = np.clip(theta_idx, 0, theta_bins - 1)
    z_idx = np.clip(z_idx, 0, z_bins - 1)

    # Get expected wall radius for each point
    r_wall = envelope[theta_idx, z_idx]

    # Wall = points near the envelope
    wall_mask = np.abs(r - r_wall) < epsilon

    return wall_mask, r_wall


def plot_cylindrical_analysis(r, theta, z, envelope, theta_centers, z_centers, output_dir):
    """Plot cylindrical analysis results."""

    # 1. Unwrapped view: θ vs z, colored by r
    fig, ax = plt.subplots(figsize=(16, 8))
    sample_idx = np.random.choice(len(r), min(100000, len(r)), replace=False)
    scatter = ax.scatter(np.degrees(theta[sample_idx]), z[sample_idx],
                        c=r[sample_idx], cmap='viridis', s=0.1, alpha=0.5)
    plt.colorbar(scatter, label='Radius (m)')
    ax.set_xlabel('θ (degrees)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Cylindrical Unwrap: θ vs Z, colored by radius')
    plt.tight_layout()
    plt.savefig(str(output_dir / "01_cylindrical_unwrap.png"), dpi=150)
    plt.close()
    print(f"Saved: 01_cylindrical_unwrap.png")

    # 2. Radial envelope heatmap
    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(envelope.T, origin='lower', aspect='auto',
                   extent=[0, 360, z_centers.min(), z_centers.max()],
                   cmap='viridis')
    plt.colorbar(im, label='Wall radius (m)')
    ax.set_xlabel('θ (degrees)')
    ax.set_ylabel('Z (m)')
    ax.set_title(f'Radial Envelope r_wall(θ, z)\nWall detected at upper {95}th percentile of radius')
    plt.tight_layout()
    plt.savefig(str(output_dir / "02_radial_envelope.png"), dpi=150)
    plt.close()
    print(f"Saved: 02_radial_envelope.png")

    # 3. Radius histogram with envelope stats
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(r, bins=100, edgecolor='black', alpha=0.7)
    envelope_mean = np.nanmean(envelope)
    ax.axvline(envelope_mean, color='r', linestyle='--', linewidth=2,
               label=f'Mean envelope: {envelope_mean:.2f}m')
    ax.set_xlabel('Radius (m)')
    ax.set_ylabel('Point Count')
    ax.set_title('Radial Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_dir / "03_radius_histogram.png"), dpi=150)
    plt.close()
    print(f"Saved: 03_radius_histogram.png")


def plot_results(points, wall_mask, interior_mask, center, output_dir):
    """Plot wall and interior results."""

    # XY views
    for mask, name, color, title in [
        (wall_mask, "wall", "gray", "WALL (outer envelope)"),
        (interior_mask, "interior", "steelblue", "INTERIOR (structure)")
    ]:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.scatter(points[mask, 0], points[mask, 1], s=0.1, alpha=0.5, c=color)
        ax.plot(center[0], center[1], 'r+', markersize=15, markeredgewidth=2)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{title}\n{np.sum(mask):,} points ({np.sum(mask)/len(points)*100:.1f}%)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(output_dir / f"{name}_xy.png"), dpi=150)
        plt.close()
        print(f"Saved: {name}_xy.png")

        # XZ view
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.scatter(points[mask, 0], points[mask, 2], s=0.1, alpha=0.5, c=color)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(f'{title} - Side View')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(output_dir / f"{name}_xz.png"), dpi=150)
        plt.close()
        print(f"Saved: {name}_xz.png")

    # Combined
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.scatter(points[wall_mask, 0], points[wall_mask, 1], s=0.1, alpha=0.3, c='gray', label=f'Wall ({np.sum(wall_mask):,})')
    ax.scatter(points[interior_mask, 0], points[interior_mask, 1], s=0.1, alpha=0.5, c='blue', label=f'Interior ({np.sum(interior_mask):,})')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Wall vs Interior (Cylindrical Envelope Method)')
    ax.set_aspect('equal')
    ax.legend(markerscale=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_dir / "combined_xy.png"), dpi=150)
    plt.close()
    print(f"Saved: combined_xy.png")


def save_las(filepath, pts):
    new_las = laspy.create(point_format=0, file_version="1.4")
    new_las.x = pts[:, 0]
    new_las.y = pts[:, 1]
    new_las.z = pts[:, 2]
    new_las.write(filepath)
    print(f"Saved: {filepath} ({len(pts):,} points)")


def main():
    input_path = "input_data/Navvis5mmShaft14_middle_10m.las"
    output_dir = Path("output/step1_wall_removal_v6_cylindrical")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1 v6: CYLINDRICAL UNWRAP + RADIAL ENVELOPE")
    print("=" * 60)

    # Load data
    print("\nLoading point cloud...")
    las = laspy.read(input_path)
    points = np.vstack((las.x, las.y, las.z)).T
    print(f"Loaded {len(points):,} points")

    # Step 1: Estimate shaft axis
    print("\n[1] Estimating shaft axis via PCA...")
    center, axis = estimate_shaft_axis(points)
    print(f"  Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    print(f"  Axis direction: ({axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f})")
    print(f"  Axis is {abs(axis[2])*100:.1f}% aligned with Z")

    # Step 2: Transform to cylindrical coordinates
    print("\n[2] Transforming to cylindrical coordinates...")
    r, theta, z = to_cylindrical(points, center, axis)
    print(f"  Radius range: {r.min():.3f}m - {r.max():.3f}m")
    print(f"  Z range: {z.min():.3f}m - {z.max():.3f}m")

    # Step 3: Compute radial envelope
    print("\n[3] Computing radial envelope r_wall(θ, z)...")
    THETA_BINS = 180  # 2° resolution
    Z_BINS = 100      # 10cm resolution for 10m
    PERCENTILE = 95   # Upper 95th percentile = wall

    envelope, theta_edges, z_edges, theta_centers, z_centers = compute_radial_envelope(
        r, theta, z,
        theta_bins=THETA_BINS,
        z_bins=Z_BINS,
        percentile=PERCENTILE
    )
    print(f"  Envelope shape: {envelope.shape}")
    print(f"  Envelope radius: {np.nanmin(envelope):.3f}m - {np.nanmax(envelope):.3f}m")
    print(f"  Mean wall radius: {np.nanmean(envelope):.3f}m")

    # Step 4: Classify wall
    print("\n[4] Classifying wall points...")
    EPSILON = 0.15  # Points within 15cm of envelope = wall

    wall_mask, r_wall = classify_wall(r, theta, z, envelope, theta_edges, z_edges, epsilon=EPSILON)
    interior_mask = ~wall_mask

    print(f"  Epsilon: {EPSILON}m")
    print(f"  WALL: {np.sum(wall_mask):,} points ({np.sum(wall_mask)/len(points)*100:.1f}%)")
    print(f"  INTERIOR: {np.sum(interior_mask):,} points ({np.sum(interior_mask)/len(points)*100:.1f}%)")

    # Test different epsilon values
    print("\n  Testing different epsilon values:")
    for eps in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        wm, _ = classify_wall(r, theta, z, envelope, theta_edges, z_edges, epsilon=eps)
        print(f"    ε={eps:.2f}m: Wall={np.sum(wm):,} ({np.sum(wm)/len(points)*100:.1f}%), Interior={np.sum(~wm):,} ({np.sum(~wm)/len(points)*100:.1f}%)")

    # Generate visualizations
    print("\n--- Generating visualizations ---")
    plot_cylindrical_analysis(r, theta, z, envelope, theta_centers, z_centers, output_dir)
    plot_results(points, wall_mask, interior_mask, center, output_dir)

    # Save LAS files
    print("\n--- Saving LAS files ---")
    save_las(str(output_dir / "wall.las"), points[wall_mask])
    save_las(str(output_dir / "interior.las"), points[interior_mask])

    # Save parameters
    np.savez(str(output_dir / "cylindrical_params.npz"),
             center=center, axis=axis,
             r=r, theta=theta, z=z,
             envelope=envelope,
             theta_edges=theta_edges, z_edges=z_edges,
             wall_mask=wall_mask)
    print(f"Saved: cylindrical_params.npz")

    print("\n" + "=" * 60)
    print("STEP 1 v6 COMPLETE")
    print("=" * 60)
    print(f"\nWALL: {np.sum(wall_mask):,} points ({np.sum(wall_mask)/len(points)*100:.1f}%)")
    print(f"INTERIOR: {np.sum(interior_mask):,} points ({np.sum(interior_mask)/len(points)*100:.1f}%)")
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
