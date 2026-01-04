"""
Wall removal module - separates cylindrical shaft wall from interior structure.
"""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (WALL_RADIUS_THRESHOLD, SHAFT_CENTER_AUTO, SHAFT_CENTER_XY,
                    LABEL_WALL)


def compute_shaft_center(points: np.ndarray) -> np.ndarray:
    """
    Compute the shaft centerline in XY plane.

    For a cylindrical shaft, the center is approximated by the centroid
    of all points in XY. For more accuracy, could use circle fitting.

    Args:
        points: Nx3 array of XYZ coordinates

    Returns:
        center_xy: 2D array [x, y] of shaft center
    """
    if SHAFT_CENTER_AUTO:
        center_xy = np.mean(points[:, :2], axis=0)
    else:
        center_xy = np.array(SHAFT_CENTER_XY)

    return center_xy


def compute_radial_distance(points: np.ndarray, center_xy: np.ndarray) -> np.ndarray:
    """
    Compute radial distance of each point from shaft center axis.

    Args:
        points: Nx3 array of XYZ coordinates
        center_xy: 2D shaft center [x, y]

    Returns:
        radii: N array of radial distances
    """
    xy_offset = points[:, :2] - center_xy
    radii = np.linalg.norm(xy_offset, axis=1)
    return radii


def remove_wall(points: np.ndarray, radius_threshold: float = None) -> tuple:
    """
    Separate wall points from interior structure based on radial distance.

    The mine shaft wall forms a cylindrical surface at the outer radius.
    Points beyond the threshold are classified as wall.

    Args:
        points: Nx3 array of XYZ coordinates
        radius_threshold: Override config threshold if specified

    Returns:
        wall_mask: Boolean mask where True = wall point
        interior_mask: Boolean mask where True = interior point
        center_xy: Computed shaft center
        radii: Radial distances for all points
    """
    if radius_threshold is None:
        radius_threshold = WALL_RADIUS_THRESHOLD

    # Compute shaft center
    center_xy = compute_shaft_center(points)
    print(f"Shaft center: ({center_xy[0]:.3f}, {center_xy[1]:.3f})")

    # Compute radial distances
    radii = compute_radial_distance(points, center_xy)

    # Classify
    wall_mask = radii > radius_threshold
    interior_mask = ~wall_mask

    n_wall = np.sum(wall_mask)
    n_interior = np.sum(interior_mask)
    print(f"Wall removal (r > {radius_threshold:.2f}m):")
    print(f"  Wall points: {n_wall:,} ({n_wall/len(points)*100:.1f}%)")
    print(f"  Interior points: {n_interior:,} ({n_interior/len(points)*100:.1f}%)")

    return wall_mask, interior_mask, center_xy, radii


def analyze_radial_distribution(radii: np.ndarray, n_bins: int = 20):
    """
    Analyze radial distribution to help tune wall threshold.

    Args:
        radii: Array of radial distances
        n_bins: Number of histogram bins
    """
    print("\n=== Radial Distribution Analysis ===")
    hist, bins = np.histogram(radii, bins=n_bins)

    # Find potential wall threshold (look for density jump)
    densities = hist / (np.pi * (bins[1:]**2 - bins[:-1]**2))

    for i in range(len(hist)):
        bar = '#' * int(hist[i] / max(hist) * 30)
        print(f"  {bins[i]:.2f}-{bins[i+1]:.2f}m: {bar} ({hist[i]:,})")

    # Suggest threshold based on density analysis
    max_density_idx = np.argmax(densities)
    suggested_threshold = bins[max_density_idx]
    print(f"\nSuggested wall threshold: ~{suggested_threshold:.2f}m")
    print(f"(based on peak point density)")
