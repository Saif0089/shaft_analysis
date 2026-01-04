"""
Platform/grating detector - identifies horizontal planar surfaces.
"""

import numpy as np
from scipy.spatial import KDTree
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (PLANARITY_THRESHOLD, PLATFORM_NORMAL_THRESHOLD,
                    PLATFORM_MAX_THICKNESS, PLATFORM_MIN_AREA,
                    REGION_GROW_RADIUS, MIN_CLUSTER_POINTS, LABEL_PLATFORM)


def detect_platforms(points: np.ndarray, features: dict,
                     already_classified: np.ndarray = None) -> tuple:
    """
    Detect platforms and gratings.

    Platforms are characterized by:
    1. High planarity (planar structure)
    2. Normal vector aligned with Z-axis (horizontal surface)
    3. Thin Z-extent (thickness)
    4. Sufficient XY area

    Args:
        points: Nx3 array of XYZ coordinates
        features: Output from compute_local_pca
        already_classified: Boolean mask of points already assigned

    Returns:
        platform_mask: Boolean mask where True = platform point
        cluster_labels: Integer labels for individual platforms
        z_levels: Dictionary mapping cluster label to Z height
    """
    n_points = len(points)
    Z_AXIS = np.array([0, 0, 1])

    if already_classified is None:
        already_classified = np.zeros(n_points, dtype=bool)

    # Step 1: Filter by planarity
    planar_mask = features['planarity'] > PLANARITY_THRESHOLD

    # Step 2: Filter by normal direction (horizontal surface = vertical normal)
    normals = features['normals']
    z_alignment = np.abs(np.sum(normals * Z_AXIS, axis=1))
    horizontal_surface_mask = z_alignment > PLATFORM_NORMAL_THRESHOLD

    # Step 3: Exclude points already classified as linear structures
    # (prevents stealing points from beams)
    not_linear = features['linearity'] < PLANARITY_THRESHOLD

    # Combine criteria
    candidate_mask = (planar_mask & horizontal_surface_mask &
                      not_linear & ~already_classified)

    n_candidates = np.sum(candidate_mask)
    print(f"Platform detection: {n_candidates:,} candidate points")

    if n_candidates == 0:
        return np.zeros(n_points, dtype=bool), np.full(n_points, -1), {}

    candidate_indices = np.where(candidate_mask)[0]
    candidate_points = points[candidate_indices]

    # Step 4: Group by Z-level first (platforms at discrete heights)
    z_groups = group_by_z_level(candidate_points, tolerance=PLATFORM_MAX_THICKNESS)

    # Step 5: Within each Z-level, region grow in XY
    cluster_labels_local = np.full(len(candidate_indices), -1, dtype=np.int32)
    z_levels = {}

    global_cluster_id = 0

    for z_level, local_indices in z_groups.items():
        if len(local_indices) < MIN_CLUSTER_POINTS:
            continue

        level_points = candidate_points[local_indices]

        # Region grow in XY plane (using 2D distances)
        local_clusters = region_grow_2d(
            level_points,
            radius=REGION_GROW_RADIUS * 2,  # Wider for platforms
            min_points=MIN_CLUSTER_POINTS
        )

        # Validate by area and thickness
        valid_clusters = validate_platform_clusters(
            level_points, local_clusters,
            min_area=PLATFORM_MIN_AREA,
            max_thickness=PLATFORM_MAX_THICKNESS
        )

        # Assign labels
        for local_label in valid_clusters:
            mask = local_clusters == local_label
            for i, level_i in enumerate(local_indices):
                if mask[i]:
                    cluster_labels_local[level_i] = global_cluster_id
            z_levels[global_cluster_id] = z_level
            global_cluster_id += 1

    # Map back to full point cloud
    platform_mask = np.zeros(n_points, dtype=bool)
    cluster_labels = np.full(n_points, -1, dtype=np.int32)

    for i, idx in enumerate(candidate_indices):
        if cluster_labels_local[i] >= 0:
            platform_mask[idx] = True
            cluster_labels[idx] = cluster_labels_local[i]

    n_platforms = np.sum(platform_mask)
    n_clusters = len(set(cluster_labels[cluster_labels >= 0]))
    print(f"  Found {n_platforms:,} platform points in {n_clusters} clusters")

    if z_levels:
        print(f"  Z-levels: {sorted(z_levels.values())}")

    return platform_mask, cluster_labels, z_levels


def group_by_z_level(points: np.ndarray, tolerance: float) -> dict:
    """
    Group points by Z-level with tolerance.

    Args:
        points: Nx3 array
        tolerance: Z-tolerance for grouping

    Returns:
        Dictionary mapping Z-level (rounded) to list of point indices
    """
    z_values = points[:, 2]

    # Round Z to tolerance
    z_rounded = np.round(z_values / tolerance) * tolerance

    groups = {}
    for i, z in enumerate(z_rounded):
        z_key = round(z, 3)  # Avoid floating point issues
        if z_key not in groups:
            groups[z_key] = []
        groups[z_key].append(i)

    return groups


def region_grow_2d(points: np.ndarray, radius: float,
                   min_points: int) -> np.ndarray:
    """
    Cluster points via region growing in XY plane.

    Args:
        points: Nx3 array (only XY used for distances)
        radius: Search radius
        min_points: Minimum cluster size

    Returns:
        labels: N array of cluster labels
    """
    n = len(points)
    labels = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)

    # Build tree on XY only
    xy_points = points[:, :2]
    tree = KDTree(xy_points)
    current_label = 0

    for i in range(n):
        if visited[i]:
            continue

        cluster_points = []
        queue = [i]

        while queue:
            idx = queue.pop(0)
            if visited[idx]:
                continue

            visited[idx] = True
            cluster_points.append(idx)

            neighbors = tree.query_ball_point(xy_points[idx], radius)
            for n_idx in neighbors:
                if not visited[n_idx]:
                    queue.append(n_idx)

        if len(cluster_points) >= min_points:
            for idx in cluster_points:
                labels[idx] = current_label
            current_label += 1

    return labels


def validate_platform_clusters(points: np.ndarray, labels: np.ndarray,
                               min_area: float, max_thickness: float) -> set:
    """
    Validate platform clusters by area and thickness.

    Args:
        points: Nx3 array
        labels: Cluster labels
        min_area: Minimum XY area
        max_thickness: Maximum Z thickness

    Returns:
        valid_labels: Set of valid cluster labels
    """
    valid_labels = set()
    unique_labels = set(labels) - {-1}

    for label in unique_labels:
        mask = labels == label
        cluster_points = points[mask]

        # Check thickness
        z_thickness = cluster_points[:, 2].max() - cluster_points[:, 2].min()
        if z_thickness > max_thickness:
            continue

        # Estimate area (convex hull would be more accurate but slower)
        xy_extent = (cluster_points[:, 0].max() - cluster_points[:, 0].min()) * \
                    (cluster_points[:, 1].max() - cluster_points[:, 1].min())

        if xy_extent >= min_area:
            valid_labels.add(label)

    return valid_labels
