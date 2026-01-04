"""
Vertical member detector - identifies vertical beams, columns, and pipes.
"""

import numpy as np
from scipy.spatial import KDTree
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (LINEARITY_THRESHOLD, VERTICAL_DOT_THRESHOLD,
                    VERTICAL_MIN_HEIGHT, REGION_GROW_RADIUS,
                    MIN_CLUSTER_POINTS, LABEL_VERTICAL)


def detect_vertical_members(points: np.ndarray, features: dict,
                            already_classified: np.ndarray = None) -> tuple:
    """
    Detect vertical steel members (beams, columns, pipes).

    Vertical members are characterized by:
    1. High linearity (linear structure)
    2. Principal direction aligned with Z-axis
    3. Sufficient Z-extent (height)

    Args:
        points: Nx3 array of XYZ coordinates
        features: Output from compute_local_pca
        already_classified: Boolean mask of points already assigned (skip these)

    Returns:
        vertical_mask: Boolean mask where True = vertical member point
        cluster_labels: Integer labels for individual vertical members (-1 = not vertical)
    """
    n_points = len(points)
    Z_AXIS = np.array([0, 0, 1])

    # Start with all points as candidates
    if already_classified is None:
        already_classified = np.zeros(n_points, dtype=bool)

    # Step 1: Filter by linearity
    linear_mask = features['linearity'] > LINEARITY_THRESHOLD

    # Step 2: Filter by direction (aligned with Z)
    # Compute absolute dot product with Z-axis
    directions = features['directions']
    z_alignment = np.abs(np.sum(directions * Z_AXIS, axis=1))
    vertical_dir_mask = z_alignment > VERTICAL_DOT_THRESHOLD

    # Combine criteria
    candidate_mask = linear_mask & vertical_dir_mask & ~already_classified

    n_candidates = np.sum(candidate_mask)
    print(f"Vertical detection: {n_candidates:,} candidate points")

    if n_candidates == 0:
        return np.zeros(n_points, dtype=bool), np.full(n_points, -1)

    # Step 3: Region growing to form clusters
    candidate_indices = np.where(candidate_mask)[0]
    candidate_points = points[candidate_indices]

    cluster_labels_local = region_grow_clusters(
        candidate_points,
        radius=REGION_GROW_RADIUS,
        min_points=MIN_CLUSTER_POINTS
    )

    # Step 4: Validate clusters by Z-extent
    valid_clusters = validate_vertical_clusters(
        candidate_points, cluster_labels_local,
        min_height=VERTICAL_MIN_HEIGHT
    )

    # Map back to full point cloud
    vertical_mask = np.zeros(n_points, dtype=bool)
    cluster_labels = np.full(n_points, -1, dtype=np.int32)

    for i, idx in enumerate(candidate_indices):
        local_label = cluster_labels_local[i]
        if local_label in valid_clusters:
            vertical_mask[idx] = True
            cluster_labels[idx] = local_label

    n_vertical = np.sum(vertical_mask)
    n_clusters = len(valid_clusters)
    print(f"  Found {n_vertical:,} vertical points in {n_clusters} clusters")

    return vertical_mask, cluster_labels


def region_grow_clusters(points: np.ndarray, radius: float,
                         min_points: int) -> np.ndarray:
    """
    Cluster points via region growing.

    Args:
        points: Nx3 array
        radius: Search radius for growing
        min_points: Minimum cluster size

    Returns:
        labels: N array of cluster labels (-1 for noise)
    """
    n = len(points)
    labels = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)

    tree = KDTree(points)
    current_label = 0

    for i in range(n):
        if visited[i]:
            continue

        # Start new cluster
        cluster_points = []
        queue = [i]

        while queue:
            idx = queue.pop(0)
            if visited[idx]:
                continue

            visited[idx] = True
            cluster_points.append(idx)

            # Find neighbors
            neighbors = tree.query_ball_point(points[idx], radius)
            for n_idx in neighbors:
                if not visited[n_idx]:
                    queue.append(n_idx)

        # Assign label if cluster is large enough
        if len(cluster_points) >= min_points:
            for idx in cluster_points:
                labels[idx] = current_label
            current_label += 1

    return labels


def validate_vertical_clusters(points: np.ndarray, labels: np.ndarray,
                               min_height: float) -> set:
    """
    Validate vertical clusters by checking Z-extent.

    Args:
        points: Nx3 array
        labels: Cluster labels
        min_height: Minimum Z range for valid vertical member

    Returns:
        valid_labels: Set of valid cluster labels
    """
    valid_labels = set()
    unique_labels = set(labels) - {-1}

    for label in unique_labels:
        mask = labels == label
        cluster_points = points[mask]

        z_range = cluster_points[:, 2].max() - cluster_points[:, 2].min()

        if z_range >= min_height:
            valid_labels.add(label)

    return valid_labels


def fit_vertical_cylinder(points: np.ndarray) -> tuple:
    """
    Fit a vertical cylinder to a set of points.

    For vertical members, we fit a 2D circle in XY and get the Z-extent.

    Args:
        points: Nx3 array of cluster points

    Returns:
        center_xy: Circle center [x, y]
        radius: Circle radius
        z_min, z_max: Z extent
    """
    # Simple circle fit via mean/std
    center_xy = np.mean(points[:, :2], axis=0)
    radii = np.linalg.norm(points[:, :2] - center_xy, axis=1)
    radius = np.mean(radii)

    z_min = points[:, 2].min()
    z_max = points[:, 2].max()

    return center_xy, radius, z_min, z_max
