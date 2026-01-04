"""
Horizontal member detector - identifies horizontal beams, pipes, and struts.
Uses direction-aware clustering to separate intersecting members.
"""

import numpy as np
from scipy.spatial import KDTree
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (LINEARITY_THRESHOLD, HORIZONTAL_DOT_THRESHOLD,
                    HORIZONTAL_MIN_LENGTH, HORIZONTAL_REGION_RADIUS,
                    MIN_CLUSTER_POINTS, ORIENTATION_BINS, ORIENTATION_TOLERANCE,
                    LABEL_HORIZONTAL_0, LABEL_HORIZONTAL_90)


def detect_horizontal_members(points: np.ndarray, features: dict,
                              already_classified: np.ndarray = None) -> tuple:
    """
    Detect horizontal steel members (beams, pipes, struts).

    Horizontal members are characterized by:
    1. High linearity (linear structure)
    2. Principal direction perpendicular to Z-axis
    3. Sufficient XY-extent (length)

    Key innovation: Cluster in (position + direction) space to separate
    intersecting members at T-junctions.

    Args:
        points: Nx3 array of XYZ coordinates
        features: Output from compute_local_pca
        already_classified: Boolean mask of points already assigned

    Returns:
        horizontal_mask: Boolean mask where True = horizontal member point
        orientation_labels: Labels indicating orientation (LABEL_HORIZONTAL_0 or LABEL_HORIZONTAL_90)
        cluster_labels: Integer labels for individual horizontal members
    """
    n_points = len(points)
    Z_AXIS = np.array([0, 0, 1])

    if already_classified is None:
        already_classified = np.zeros(n_points, dtype=bool)

    # Step 1: Filter by linearity
    linear_mask = features['linearity'] > LINEARITY_THRESHOLD

    # Step 2: Filter by direction (perpendicular to Z)
    directions = features['directions']
    z_alignment = np.abs(np.sum(directions * Z_AXIS, axis=1))
    horizontal_dir_mask = z_alignment < HORIZONTAL_DOT_THRESHOLD

    # Combine criteria
    candidate_mask = linear_mask & horizontal_dir_mask & ~already_classified

    n_candidates = np.sum(candidate_mask)
    print(f"Horizontal detection: {n_candidates:,} candidate points")

    if n_candidates == 0:
        return (np.zeros(n_points, dtype=bool),
                np.full(n_points, -1),
                np.full(n_points, -1))

    candidate_indices = np.where(candidate_mask)[0]
    candidate_points = points[candidate_indices]
    candidate_directions = directions[candidate_indices]

    # Step 3: Compute XY orientation angle for each candidate
    xy_angles = compute_xy_angles(candidate_directions)

    # Step 4: Bin by orientation
    orientation_bins = bin_by_orientation(xy_angles)

    # Step 5: Cluster within each orientation bin using position
    cluster_labels_local = np.full(len(candidate_indices), -1, dtype=np.int32)
    orientation_labels_local = np.full(len(candidate_indices), -1, dtype=np.int32)

    global_cluster_id = 0

    for bin_idx, bin_angle in enumerate(ORIENTATION_BINS):
        bin_mask = orientation_bins == bin_idx
        bin_indices = np.where(bin_mask)[0]

        if len(bin_indices) < MIN_CLUSTER_POINTS:
            continue

        bin_points = candidate_points[bin_indices]

        # Region grow within this orientation bin (use larger radius for sparse horizontal points)
        local_clusters = region_grow_horizontal(
            bin_points,
            radius=HORIZONTAL_REGION_RADIUS,
            min_points=min(MIN_CLUSTER_POINTS, len(bin_points) // 3)  # Adaptive min
        )

        # Validate by length
        valid_clusters = validate_horizontal_clusters(
            bin_points, local_clusters,
            min_length=HORIZONTAL_MIN_LENGTH
        )

        # Assign labels
        label = LABEL_HORIZONTAL_0 if bin_angle == 0 else LABEL_HORIZONTAL_90

        for local_label in valid_clusters:
            mask = local_clusters == local_label
            for i, bin_i in enumerate(bin_indices):
                if mask[i]:
                    cluster_labels_local[bin_i] = global_cluster_id
                    orientation_labels_local[bin_i] = label
            global_cluster_id += 1

    # Map back to full point cloud
    horizontal_mask = np.zeros(n_points, dtype=bool)
    orientation_labels = np.full(n_points, -1, dtype=np.int32)
    cluster_labels = np.full(n_points, -1, dtype=np.int32)

    for i, idx in enumerate(candidate_indices):
        if cluster_labels_local[i] >= 0:
            horizontal_mask[idx] = True
            orientation_labels[idx] = orientation_labels_local[i]
            cluster_labels[idx] = cluster_labels_local[i]

    n_horizontal = np.sum(horizontal_mask)
    n_h0 = np.sum(orientation_labels == LABEL_HORIZONTAL_0)
    n_h90 = np.sum(orientation_labels == LABEL_HORIZONTAL_90)
    print(f"  Found {n_horizontal:,} horizontal points")
    print(f"    0° aligned: {n_h0:,}")
    print(f"    90° aligned: {n_h90:,}")

    return horizontal_mask, orientation_labels, cluster_labels


def compute_xy_angles(directions: np.ndarray) -> np.ndarray:
    """
    Compute XY plane angle (0-180°) for each direction vector.

    We use 0-180° because direction vectors are bidirectional
    (pointing either way along a beam is equivalent).

    Args:
        directions: Nx3 direction vectors

    Returns:
        angles: N array of angles in degrees (0-180)
    """
    # Project to XY plane and normalize
    xy_dirs = directions[:, :2].copy()
    norms = np.linalg.norm(xy_dirs, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1  # Avoid division by zero
    xy_dirs = xy_dirs / norms

    # Compute angle from X-axis
    angles = np.arctan2(xy_dirs[:, 1], xy_dirs[:, 0])
    angles = np.degrees(angles)

    # Map to 0-180 range (bidirectional)
    angles = angles % 180

    return angles


def bin_by_orientation(angles: np.ndarray) -> np.ndarray:
    """
    Assign angles to orientation bins.

    Args:
        angles: N array of angles in degrees (0-180)

    Returns:
        bins: N array of bin indices
    """
    n = len(angles)
    bins = np.full(n, -1, dtype=np.int32)

    for bin_idx, bin_angle in enumerate(ORIENTATION_BINS):
        # Check if angle is within tolerance of this bin
        # Handle wraparound for 0/180
        diff = np.abs(angles - bin_angle)
        diff = np.minimum(diff, 180 - diff)  # Wraparound

        in_bin = diff <= ORIENTATION_TOLERANCE
        bins[in_bin] = bin_idx

    return bins


def region_grow_horizontal(points: np.ndarray, radius: float,
                           min_points: int) -> np.ndarray:
    """
    Cluster points via region growing for horizontal members.

    Args:
        points: Nx3 array
        radius: Search radius
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

        cluster_points = []
        queue = [i]

        while queue:
            idx = queue.pop(0)
            if visited[idx]:
                continue

            visited[idx] = True
            cluster_points.append(idx)

            neighbors = tree.query_ball_point(points[idx], radius)
            for n_idx in neighbors:
                if not visited[n_idx]:
                    queue.append(n_idx)

        if len(cluster_points) >= min_points:
            for idx in cluster_points:
                labels[idx] = current_label
            current_label += 1

    return labels


def validate_horizontal_clusters(points: np.ndarray, labels: np.ndarray,
                                 min_length: float) -> set:
    """
    Validate horizontal clusters by checking XY-extent.

    Args:
        points: Nx3 array
        labels: Cluster labels
        min_length: Minimum XY span for valid horizontal member

    Returns:
        valid_labels: Set of valid cluster labels
    """
    valid_labels = set()
    unique_labels = set(labels) - {-1}

    for label in unique_labels:
        mask = labels == label
        cluster_points = points[mask]

        # Compute XY extent (length along principal direction)
        xy_points = cluster_points[:, :2]
        if len(xy_points) < 2:
            continue

        # Use PCA to get principal direction in XY
        centered = xy_points - np.mean(xy_points, axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Project onto principal direction
        principal = eigvecs[:, np.argmax(eigvals)]
        projections = centered @ principal

        length = projections.max() - projections.min()

        if length >= min_length:
            valid_labels.add(label)

    return valid_labels
