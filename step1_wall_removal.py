#!/usr/bin/env python3
"""
Step 1: Wall Removal using Fitted Circle

Uses least-squares circle fitting to find the true shaft center,
not just the centroid which can be biased by interior structure.
"""

import numpy as np
import laspy
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from pathlib import Path


def fit_circle_least_squares(x, y):
    """
    Fit a circle to 2D points using least squares.

    Uses algebraic distance minimization with refinement.

    Args:
        x, y: Arrays of point coordinates

    Returns:
        cx, cy: Circle center
        r: Circle radius
    """
    # Initial guess using centroid and mean distance
    cx_init = np.mean(x)
    cy_init = np.mean(y)
    r_init = np.mean(np.sqrt((x - cx_init)**2 + (y - cy_init)**2))

    def residuals(params):
        cx, cy, r = params
        distances = np.sqrt((x - cx)**2 + (y - cy)**2)
        return distances - r

    # Optimize
    result = least_squares(residuals, [cx_init, cy_init, r_init], method='lm')
    cx, cy, r = result.x

    return cx, cy, r


def fit_circle_to_outer_points(points, percentile=90):
    """
    Fit circle to the outer shell of points (the wall).

    Only uses points beyond a certain radial percentile to avoid
    interior structure biasing the fit.

    Args:
        points: Nx3 array
        percentile: Use points beyond this radial percentile

    Returns:
        cx, cy: Fitted circle center
        r: Fitted circle radius
    """
    x, y = points[:, 0], points[:, 1]

    # Initial centroid
    cx_init = np.mean(x)
    cy_init = np.mean(y)

    # Compute radial distances from initial center
    radii = np.sqrt((x - cx_init)**2 + (y - cy_init)**2)

    # Select outer points (likely wall)
    threshold = np.percentile(radii, percentile)
    outer_mask = radii > threshold

    print(f"Using {np.sum(outer_mask):,} outer points (>{percentile}th percentile) for circle fit")

    # Fit circle to outer points
    cx, cy, r = fit_circle_least_squares(x[outer_mask], y[outer_mask])

    return cx, cy, r, outer_mask


def remove_wall(points, cx, cy, wall_radius, tolerance=0.1):
    """
    Remove wall points based on fitted circle.

    Args:
        points: Nx3 array
        cx, cy: Circle center
        wall_radius: Fitted wall radius
        tolerance: Points within (wall_radius - tolerance) are interior

    Returns:
        wall_mask: Boolean mask for wall points
        interior_mask: Boolean mask for interior points
    """
    x, y = points[:, 0], points[:, 1]
    radii = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Interior = points closer to center than wall
    interior_threshold = wall_radius - tolerance
    interior_mask = radii < interior_threshold
    wall_mask = ~interior_mask

    return wall_mask, interior_mask, radii


def plot_xy_view(points, mask, title, filename, cx=None, cy=None, r=None,
                 color_by_radius=False, radii=None):
    """Generate top-down XY view."""
    fig, ax = plt.subplots(figsize=(12, 12))

    selected = points[mask]

    if color_by_radius and radii is not None:
        scatter = ax.scatter(selected[:, 0], selected[:, 1],
                            c=radii[mask], cmap='viridis', s=0.1, alpha=0.5)
        plt.colorbar(scatter, label='Radial distance (m)')
    else:
        ax.scatter(selected[:, 0], selected[:, 1], s=0.1, alpha=0.5, c='steelblue')

    # Draw fitted circle if provided
    if cx is not None and cy is not None and r is not None:
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = cx + r * np.cos(theta)
        circle_y = cy + r * np.sin(theta)
        ax.plot(circle_x, circle_y, 'r-', linewidth=2, label=f'Fitted circle (r={r:.3f}m)')
        ax.plot(cx, cy, 'r+', markersize=15, markeredgewidth=2, label=f'Center ({cx:.3f}, {cy:.3f})')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_xz_view(points, mask, title, filename):
    """Generate side XZ view."""
    fig, ax = plt.subplots(figsize=(12, 8))

    selected = points[mask]
    ax.scatter(selected[:, 0], selected[:, 2], s=0.1, alpha=0.5, c='steelblue')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def save_las(filepath, points, source_las=None):
    """Save points to LAS file."""
    new_las = laspy.create(point_format=0, file_version="1.4")
    new_las.x = points[:, 0]
    new_las.y = points[:, 1]
    new_las.z = points[:, 2]
    new_las.write(filepath)
    print(f"Saved: {filepath} ({len(points):,} points)")


def main():
    input_path = "../Navvis5mmShaft14_middle_10m.las"
    output_dir = Path("output/step1_wall_removal")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1: WALL REMOVAL USING FITTED CIRCLE")
    print("=" * 60)

    # Load data
    print("\nLoading point cloud...")
    las = laspy.read(input_path)
    points = np.vstack((las.x, las.y, las.z)).T
    print(f"Loaded {len(points):,} points")

    # Compute centroid (for comparison)
    centroid_x = np.mean(points[:, 0])
    centroid_y = np.mean(points[:, 1])
    print(f"\nCentroid (biased by interior): ({centroid_x:.3f}, {centroid_y:.3f})")

    # Fit circle to outer points (wall)
    print("\nFitting circle to outer points...")
    cx, cy, r, outer_mask = fit_circle_to_outer_points(points, percentile=85)
    print(f"Fitted circle center: ({cx:.3f}, {cy:.3f})")
    print(f"Fitted circle radius: {r:.3f}m")
    print(f"Center offset from centroid: ({cx - centroid_x:.3f}, {cy - centroid_y:.3f})")

    # Remove wall with different tolerances to find best split
    print("\n--- Trying different wall tolerances ---")
    for tol in [0.05, 0.1, 0.15, 0.2, 0.3]:
        wall_mask, interior_mask, radii = remove_wall(points, cx, cy, r, tolerance=tol)
        print(f"  Tolerance {tol:.2f}m: Wall={np.sum(wall_mask):,} ({np.sum(wall_mask)/len(points)*100:.1f}%), "
              f"Interior={np.sum(interior_mask):,} ({np.sum(interior_mask)/len(points)*100:.1f}%)")

    # Use 0.15m tolerance as default (can adjust)
    TOLERANCE = 0.15
    print(f"\n--- Using tolerance = {TOLERANCE}m ---")
    wall_mask, interior_mask, radii = remove_wall(points, cx, cy, r, tolerance=TOLERANCE)

    wall_points = points[wall_mask]
    interior_points = points[interior_mask]

    print(f"\nWall points: {len(wall_points):,} ({len(wall_points)/len(points)*100:.1f}%)")
    print(f"Interior points: {len(interior_points):,} ({len(interior_points)/len(points)*100:.1f}%)")

    # Generate visualizations
    print("\n--- Generating visualizations ---")

    # 1. All points with fitted circle
    plot_xy_view(points, np.ones(len(points), dtype=bool),
                 f"All Points (n={len(points):,}) with Fitted Circle",
                 str(output_dir / "01_all_points_xy.png"),
                 cx, cy, r, color_by_radius=True, radii=radii)

    # 2. Wall points only (top view)
    plot_xy_view(points, wall_mask,
                 f"WALL Points (n={len(wall_points):,})",
                 str(output_dir / "02_wall_xy.png"),
                 cx, cy, r)

    # 3. Wall points (side view)
    plot_xz_view(points, wall_mask,
                 f"WALL Points - Side View (n={len(wall_points):,})",
                 str(output_dir / "03_wall_xz.png"))

    # 4. Interior points only (top view)
    plot_xy_view(points, interior_mask,
                 f"INTERIOR Points (n={len(interior_points):,})",
                 str(output_dir / "04_interior_xy.png"),
                 cx, cy, r)

    # 5. Interior points (side view)
    plot_xz_view(points, interior_mask,
                 f"INTERIOR Points - Side View (n={len(interior_points):,})",
                 str(output_dir / "05_interior_xz.png"))

    # 6. Radial distribution histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(radii, bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(r - TOLERANCE, color='r', linestyle='--', linewidth=2,
               label=f'Interior threshold (r-{TOLERANCE}m = {r-TOLERANCE:.2f}m)')
    ax.axvline(r, color='g', linestyle='-', linewidth=2,
               label=f'Fitted wall radius ({r:.2f}m)')
    ax.set_xlabel('Radial Distance from Fitted Center (m)')
    ax.set_ylabel('Point Count')
    ax.set_title('Radial Distribution of Points')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_dir / "06_radial_histogram.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / '06_radial_histogram.png'}")

    # Save LAS files
    print("\n--- Saving LAS files ---")
    save_las(str(output_dir / "wall.las"), wall_points)
    save_las(str(output_dir / "interior.las"), interior_points)

    # Save parameters for next step
    np.savez(str(output_dir / "wall_params.npz"),
             cx=cx, cy=cy, r=r, tolerance=TOLERANCE,
             wall_mask=wall_mask, interior_mask=interior_mask)
    print(f"Saved: {output_dir / 'wall_params.npz'}")

    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("Images generated:")
    print("  - 01_all_points_xy.png     : All points with fitted circle")
    print("  - 02_wall_xy.png           : Wall points (top view)")
    print("  - 03_wall_xz.png           : Wall points (side view)")
    print("  - 04_interior_xy.png       : Interior points (top view)")
    print("  - 05_interior_xz.png       : Interior points (side view)")
    print("  - 06_radial_histogram.png  : Radial distribution")


if __name__ == "__main__":
    main()
