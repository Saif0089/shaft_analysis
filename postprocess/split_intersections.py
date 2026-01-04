"""
Post-processing module for splitting intersecting members at T-junctions.
"""

import numpy as np
from scipy.spatial import KDTree
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MIN_CLUSTER_POINTS


def split_intersections(points: np.ndarray, cluster_labels: np.ndarray,
                        features: dict, direction_variance_threshold: float = 0.3) -> np.ndarray:
    """
    Split clusters at T-junction intersections based on direction variance.

    When two horizontal members intersect in a T or + shape, they may get
    merged if they're close enough. This function detects such cases by
    looking for spikes in local direction variance along the cluster.

    Args:
        points: Nx3 array of XYZ coordinates
        cluster_labels: Current cluster assignments
        features: PCA features including directions
        direction_variance_threshold: Threshold for detecting splits

    Returns:
        new_labels: Updated cluster labels with splits applied
    """
    new_labels = cluster_labels.copy()
    directions = features['directions']

    unique_labels = set(cluster_labels) - {-1}
    next_label = max(cluster_labels) + 1

    for label in unique_labels:
        mask = cluster_labels == label
        cluster_indices = np.where(mask)[0]

        if len(cluster_indices) < MIN_CLUSTER_POINTS * 2:
            continue  # Too small to split

        cluster_points = points[cluster_indices]
        cluster_directions = directions[cluster_indices]

        # Find split points
        split_indices = find_direction_variance_spikes(
            cluster_points, cluster_directions,
            threshold=direction_variance_threshold
        )

        if len(split_indices) == 0:
            continue

        # Split the cluster
        sub_labels = split_cluster_at_points(
            cluster_points, split_indices
        )

        # Remap sub-labels to global labels
        unique_sub = set(sub_labels) - {-1}
        for i, sub_label in enumerate(sorted(unique_sub)):
            sub_mask = sub_labels == sub_label
            global_indices = cluster_indices[sub_mask]

            if i == 0:
                # Keep original label for first sub-cluster
                pass
            else:
                # Assign new label
                new_labels[global_indices] = next_label
                next_label += 1

    n_splits = next_label - max(cluster_labels) - 1
    if n_splits > 0:
        print(f"Split {n_splits} clusters at intersections")

    return new_labels


def find_direction_variance_spikes(points: np.ndarray, directions: np.ndarray,
                                   threshold: float, window_size: int = 10) -> list:
    """
    Find points where direction variance spikes (indicating intersection).

    Args:
        points: Nx3 cluster points
        directions: Nx3 direction vectors
        threshold: Variance threshold for spike detection
        window_size: Number of neighbors for local variance

    Returns:
        List of indices where splits should occur
    """
    n = len(points)
    if n < window_size * 2:
        return []

    # Build KDTree for neighbor queries
    tree = KDTree(points)

    # Compute local direction variance for each point
    variances = np.zeros(n)

    for i in range(n):
        _, neighbor_idx = tree.query(points[i], k=min(window_size, n))
        neighbor_dirs = directions[neighbor_idx]

        # Compute variance of direction vectors
        # Use angular spread as variance measure
        mean_dir = np.mean(neighbor_dirs, axis=0)
        mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-6)

        # Angular deviation from mean
        dots = np.abs(np.sum(neighbor_dirs * mean_dir, axis=1))
        variance = 1 - np.mean(dots)  # High if directions are diverse

        variances[i] = variance

    # Find spike points (local maxima above threshold)
    spike_indices = []

    for i in range(1, n - 1):
        if variances[i] > threshold:
            # Check if local maximum
            if variances[i] > variances[i-1] and variances[i] > variances[i+1]:
                spike_indices.append(i)

    return spike_indices


def split_cluster_at_points(points: np.ndarray, split_indices: list) -> np.ndarray:
    """
    Split a cluster at the given indices using region growing.

    Args:
        points: Nx3 cluster points
        split_indices: Indices where splits should occur

    Returns:
        sub_labels: N array of sub-cluster labels
    """
    n = len(points)
    sub_labels = np.full(n, -1, dtype=np.int32)

    # Mark split points as barriers
    is_barrier = np.zeros(n, dtype=bool)
    for idx in split_indices:
        is_barrier[idx] = True

    # Region grow from non-barrier points
    tree = KDTree(points)
    visited = np.zeros(n, dtype=bool)
    current_label = 0

    # Use mean distance to neighbors as radius
    distances, _ = tree.query(points, k=6)
    radius = np.mean(distances[:, 1:]) * 2

    for start_idx in range(n):
        if visited[start_idx] or is_barrier[start_idx]:
            continue

        # Grow region
        cluster_points = []
        queue = [start_idx]

        while queue:
            idx = queue.pop(0)
            if visited[idx] or is_barrier[idx]:
                continue

            visited[idx] = True
            cluster_points.append(idx)
            sub_labels[idx] = current_label

            neighbors = tree.query_ball_point(points[idx], radius)
            for n_idx in neighbors:
                if not visited[n_idx] and not is_barrier[n_idx]:
                    queue.append(n_idx)

        if len(cluster_points) > 0:
            current_label += 1

    # Assign barrier points to nearest cluster
    for idx in split_indices:
        if sub_labels[idx] == -1:
            # Find nearest non-barrier point
            distances, indices = tree.query(points[idx], k=n)
            for i, neighbor_idx in enumerate(indices):
                if sub_labels[neighbor_idx] >= 0:
                    sub_labels[idx] = sub_labels[neighbor_idx]
                    break

    return sub_labels


def merge_collinear_segments(points: np.ndarray, cluster_labels: np.ndarray,
                             features: dict, angle_threshold: float = 10.0,
                             distance_threshold: float = 0.2) -> np.ndarray:
    """
    Merge segments that are collinear (on the same line).

    Args:
        points: Nx3 array
        cluster_labels: Cluster assignments
        features: PCA features
        angle_threshold: Maximum angle difference (degrees) for merging
        distance_threshold: Maximum gap between segment endpoints

    Returns:
        merged_labels: Updated cluster labels
    """
    directions = features['directions']
    merged_labels = cluster_labels.copy()

    unique_labels = sorted(set(cluster_labels) - {-1})

    # Compute cluster statistics
    cluster_stats = {}
    for label in unique_labels:
        mask = cluster_labels == label
        cluster_points = points[mask]
        cluster_dirs = directions[mask]

        # Mean direction
        mean_dir = np.mean(cluster_dirs, axis=0)
        mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-6)

        # Centroid and extent
        centroid = np.mean(cluster_points, axis=0)

        # Endpoints (approximate)
        projections = cluster_points @ mean_dir
        min_idx = np.argmin(projections)
        max_idx = np.argmax(projections)
        endpoint1 = cluster_points[min_idx]
        endpoint2 = cluster_points[max_idx]

        cluster_stats[label] = {
            'direction': mean_dir,
            'centroid': centroid,
            'endpoints': (endpoint1, endpoint2)
        }

    # Find merge candidates
    merged = set()
    merge_map = {l: l for l in unique_labels}

    for i, label1 in enumerate(unique_labels):
        if label1 in merged:
            continue

        stats1 = cluster_stats[label1]

        for label2 in unique_labels[i+1:]:
            if label2 in merged:
                continue

            stats2 = cluster_stats[label2]

            # Check direction similarity
            dot = np.abs(np.dot(stats1['direction'], stats2['direction']))
            angle = np.degrees(np.arccos(np.clip(dot, -1, 1)))

            if angle > angle_threshold:
                continue

            # Check endpoint proximity
            endpoints1 = stats1['endpoints']
            endpoints2 = stats2['endpoints']

            min_dist = min(
                np.linalg.norm(endpoints1[0] - endpoints2[0]),
                np.linalg.norm(endpoints1[0] - endpoints2[1]),
                np.linalg.norm(endpoints1[1] - endpoints2[0]),
                np.linalg.norm(endpoints1[1] - endpoints2[1])
            )

            if min_dist < distance_threshold:
                # Merge label2 into label1
                merge_map[label2] = merge_map[label1]
                merged.add(label2)

    # Apply merge map
    for old_label, new_label in merge_map.items():
        if old_label != new_label:
            merged_labels[cluster_labels == old_label] = new_label

    n_merges = len(merged)
    if n_merges > 0:
        print(f"Merged {n_merges} collinear segment pairs")

    return merged_labels
