#!/usr/bin/env python3
"""
Step 2: Extract Pipes/Cables and Separate from Steel Structure

Input: output/step1_wall_stripped/interior.las
Output: output/step2_pipes_extracted/
    - pipes.las (vertical cylindrical elements)
    - steel_structure.las (buntons, beams - what's left)
    - platforms.las (horizontal planar elements)
    - noise.las (sparse/random points)

Strategy:
1. Voxelize + local PCA for feature extraction
2. Identify pipe candidates (linear, vertical, small cross-section)
3. Z-tracking: slice along Z, track clusters, keep long continuous tracks
4. Remove platforms (planar, horizontal, repeated Z-levels)
5. Remaining = steel structure
"""

import numpy as np
import laspy
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.ndimage import label
from pathlib import Path
from collections import defaultdict


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


def compute_local_pca(points, k=20):
    """
    Compute local PCA features for each point.

    Returns:
        linearity, planarity, scattering: shape descriptors
        directions: principal direction (eigenvector of λ1)
        normals: surface normal (eigenvector of λ3)
    """
    n = len(points)
    print(f"Computing local PCA for {n:,} points (k={k})...")

    tree = KDTree(points)
    _, indices = tree.query(points, k=k+1)

    linearity = np.zeros(n)
    planarity = np.zeros(n)
    scattering = np.zeros(n)
    directions = np.zeros((n, 3))
    normals = np.zeros((n, 3))

    for i in range(n):
        if i % 50000 == 0 and i > 0:
            print(f"  {i/n*100:.0f}%...")

        neighbors = points[indices[i, 1:]]
        centered = neighbors - np.mean(neighbors, axis=0)
        cov = np.cov(centered.T)

        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[idx], 1e-10)
        eigvecs = eigvecs[:, idx]

        l1, l2, l3 = eigvals
        linearity[i] = (l1 - l2) / l1
        planarity[i] = (l2 - l3) / l1
        scattering[i] = l3 / l1
        directions[i] = eigvecs[:, 0]
        normals[i] = eigvecs[:, 2]

    print("  Done.")
    return linearity, planarity, scattering, directions, normals


def identify_pipe_candidates(points, linearity, directions,
                             linearity_thresh=0.6,
                             vertical_thresh=0.85):
    """
    Identify pipe/cable candidates based on local geometry.

    Pipes are:
    - Strongly linear (high linearity)
    - Direction mostly vertical (aligned with Z)
    """
    Z_AXIS = np.array([0, 0, 1])

    # Linear structures
    is_linear = linearity > linearity_thresh

    # Vertical direction
    z_alignment = np.abs(np.sum(directions * Z_AXIS, axis=1))
    is_vertical = z_alignment > vertical_thresh

    # Pipe candidates = linear AND vertical
    pipe_candidates = is_linear & is_vertical

    print(f"Pipe candidates: {np.sum(pipe_candidates):,} points")
    print(f"  (linear: {np.sum(is_linear):,}, vertical: {np.sum(is_vertical):,})")

    return pipe_candidates


def identify_platforms(points, planarity, normals,
                       planarity_thresh=0.5,
                       horizontal_thresh=0.85):
    """
    Identify platform/grating candidates.

    Platforms are:
    - Planar (high planarity)
    - Horizontal surface (normal points up/down)
    """
    Z_AXIS = np.array([0, 0, 1])

    # Planar structures
    is_planar = planarity > planarity_thresh

    # Horizontal surface (normal aligned with Z)
    normal_z = np.abs(np.sum(normals * Z_AXIS, axis=1))
    is_horizontal = normal_z > horizontal_thresh

    # Platform candidates = planar AND horizontal
    platform_candidates = is_planar & is_horizontal

    print(f"Platform candidates: {np.sum(platform_candidates):,} points")

    return platform_candidates


def z_tracking(points, candidate_mask,
               slice_thickness=0.25,
               xy_distance_thresh=0.3,
               min_track_length=2.0):
    """
    Track candidates along Z to filter by continuity.

    Only keep candidates that form long continuous tracks.
    Platforms/noise appear in few slices and get filtered out.

    Returns:
        valid_mask: boolean mask of points belonging to long tracks
        track_labels: integer label for each point's track (-1 if not in track)
    """
    if np.sum(candidate_mask) == 0:
        return np.zeros(len(points), dtype=bool), np.full(len(points), -1)

    candidate_indices = np.where(candidate_mask)[0]
    candidate_points = points[candidate_indices]

    z_min, z_max = candidate_points[:, 2].min(), candidate_points[:, 2].max()
    n_slices = int(np.ceil((z_max - z_min) / slice_thickness))

    print(f"Z-tracking: {n_slices} slices of {slice_thickness}m each")

    # Assign points to Z slices
    slice_idx = ((candidate_points[:, 2] - z_min) / slice_thickness).astype(int)
    slice_idx = np.clip(slice_idx, 0, n_slices - 1)

    # Build slice -> points mapping
    slice_points = defaultdict(list)
    for i, si in enumerate(slice_idx):
        slice_points[si].append(i)

    # Cluster within each slice (simple XY distance clustering)
    slice_clusters = {}
    for si in range(n_slices):
        if si not in slice_points or len(slice_points[si]) < 3:
            slice_clusters[si] = []
            continue

        pts_idx = np.array(slice_points[si])
        pts_xy = candidate_points[pts_idx, :2]

        # Simple clustering: group points within xy_distance_thresh
        if len(pts_xy) < 2:
            slice_clusters[si] = [(pts_idx, np.mean(pts_xy, axis=0))]
            continue

        tree = KDTree(pts_xy)
        visited = np.zeros(len(pts_xy), dtype=bool)
        clusters = []

        for j in range(len(pts_xy)):
            if visited[j]:
                continue

            # BFS to find cluster
            queue = [j]
            cluster_pts = []

            while queue:
                curr = queue.pop(0)
                if visited[curr]:
                    continue
                visited[curr] = True
                cluster_pts.append(curr)

                neighbors = tree.query_ball_point(pts_xy[curr], xy_distance_thresh)
                for n in neighbors:
                    if not visited[n]:
                        queue.append(n)

            if len(cluster_pts) >= 3:
                cluster_indices = pts_idx[cluster_pts]
                cluster_centroid = np.mean(pts_xy[cluster_pts], axis=0)
                clusters.append((cluster_indices, cluster_centroid))

        slice_clusters[si] = clusters

    # Link clusters across slices
    tracks = []  # Each track: list of (slice_idx, cluster_idx, point_indices)

    # Start tracks from bottom slice
    for si in range(n_slices):
        for ci, (pts_idx, centroid) in enumerate(slice_clusters[si]):
            # Check if this cluster continues an existing track
            matched = False
            for track in tracks:
                last_slice, last_ci, last_pts, last_centroid = track[-1]

                # Can only link adjacent or nearby slices
                if si - last_slice > 2:
                    continue

                # Check XY distance
                if np.linalg.norm(centroid - last_centroid) < xy_distance_thresh * 2:
                    track.append((si, ci, pts_idx, centroid))
                    matched = True
                    break

            if not matched:
                # Start new track
                tracks.append([(si, ci, pts_idx, centroid)])

    # Filter tracks by length
    valid_tracks = []
    for track in tracks:
        z_coords = [candidate_points[entry[2], 2].mean() for entry in track]
        track_length = max(z_coords) - min(z_coords)

        if track_length >= min_track_length:
            valid_tracks.append(track)

    print(f"  Found {len(tracks)} tracks, {len(valid_tracks)} valid (length >= {min_track_length}m)")

    # Build output masks
    valid_mask = np.zeros(len(points), dtype=bool)
    track_labels = np.full(len(points), -1, dtype=np.int32)

    for track_id, track in enumerate(valid_tracks):
        for entry in track:
            pts_idx = entry[2]
            original_idx = candidate_indices[pts_idx]
            valid_mask[original_idx] = True
            track_labels[original_idx] = track_id

    return valid_mask, track_labels


def filter_sparse_noise(points, mask, k=10, density_percentile=10):
    """
    Filter out sparse/noisy points based on local density.
    """
    if np.sum(mask) == 0:
        return mask

    subset_idx = np.where(mask)[0]
    subset_pts = points[subset_idx]

    tree = KDTree(subset_pts)
    distances, _ = tree.query(subset_pts, k=k+1)
    avg_dist = np.mean(distances[:, 1:], axis=1)

    # Remove points with very high average distance (sparse)
    threshold = np.percentile(avg_dist, 100 - density_percentile)
    dense_mask = avg_dist < threshold

    # Update original mask
    new_mask = mask.copy()
    new_mask[subset_idx[~dense_mask]] = False

    removed = np.sum(mask) - np.sum(new_mask)
    print(f"  Removed {removed:,} sparse points")

    return new_mask


def plot_results(points, pipe_mask, platform_mask, steel_mask, noise_mask, output_dir):
    """Generate visualization of results."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    masks = [
        (pipe_mask, "PIPES (vertical cylinders)", "red"),
        (platform_mask, "PLATFORMS", "yellow"),
        (steel_mask, "STEEL STRUCTURE", "blue"),
        (noise_mask, "NOISE/OTHER", "gray"),
    ]

    # XY views
    for i, (mask, title, color) in enumerate(masks):
        if i >= 3:
            break
        ax = axes[0, i]
        if np.sum(mask) > 0:
            ax.scatter(points[mask, 0], points[mask, 1], s=0.1, alpha=0.5, c=color)
        ax.set_title(f'{title}\n{np.sum(mask):,} pts ({np.sum(mask)/len(points)*100:.1f}%)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # XZ views
    for i, (mask, title, color) in enumerate(masks):
        if i >= 3:
            break
        ax = axes[1, i]
        if np.sum(mask) > 0:
            ax.scatter(points[mask, 0], points[mask, 2], s=0.1, alpha=0.5, c=color)
        ax.set_title(f'{title} - Side View')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_dir / "step2_results.png"), dpi=150)
    plt.close()
    print(f"Saved: step2_results.png")

    # Combined view
    fig, ax = plt.subplots(figsize=(14, 14))
    for mask, title, color in masks[::-1]:  # Draw noise first
        if np.sum(mask) > 0:
            ax.scatter(points[mask, 0], points[mask, 1], s=0.1, alpha=0.5, c=color,
                      label=f'{title} ({np.sum(mask):,})')
    ax.set_title('Step 2: Pipe Extraction Results')
    ax.set_aspect('equal')
    ax.legend(markerscale=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_dir / "step2_combined.png"), dpi=150)
    plt.close()
    print(f"Saved: step2_combined.png")


def main():
    input_path = Path("output/step1_wall_stripped/interior.las")
    output_dir = Path("output/step2_pipes_extracted")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 2: EXTRACT PIPES & SEPARATE STEEL STRUCTURE")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")

    # Load interior points from step 1
    print("\n[1/6] Loading interior points...")
    points = load_las(str(input_path))

    # Compute local PCA features
    print("\n[2/6] Computing local PCA features...")
    linearity, planarity, scattering, directions, normals = compute_local_pca(points, k=20)

    print(f"\nFeature statistics:")
    print(f"  Linearity: {linearity.min():.2f} - {linearity.max():.2f}, mean={linearity.mean():.2f}")
    print(f"  Planarity: {planarity.min():.2f} - {planarity.max():.2f}, mean={planarity.mean():.2f}")

    # Identify pipe candidates
    print("\n[3/6] Identifying pipe candidates...")
    pipe_candidates = identify_pipe_candidates(points, linearity, directions,
                                                linearity_thresh=0.6,
                                                vertical_thresh=0.85)

    # Z-tracking to filter pipes by continuity
    print("\n[4/6] Z-tracking for pipe continuity...")
    pipe_mask, pipe_tracks = z_tracking(points, pipe_candidates,
                                         slice_thickness=0.2,
                                         xy_distance_thresh=0.25,
                                         min_track_length=1.5)

    # Identify platforms
    print("\n[5/6] Identifying platforms...")
    platform_candidates = identify_platforms(points, planarity, normals,
                                              planarity_thresh=0.5,
                                              horizontal_thresh=0.85)

    # Exclude already classified as pipes
    platform_mask = platform_candidates & ~pipe_mask

    # Steel structure = not pipe, not platform, and reasonably dense
    print("\n[6/6] Separating steel structure from noise...")
    remaining_mask = ~pipe_mask & ~platform_mask

    # Filter noise from remaining (sparse points)
    steel_mask = filter_sparse_noise(points, remaining_mask, k=15, density_percentile=5)
    noise_mask = remaining_mask & ~steel_mask

    # Summary
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"PIPES:          {np.sum(pipe_mask):,} ({np.sum(pipe_mask)/len(points)*100:.1f}%)")
    print(f"PLATFORMS:      {np.sum(platform_mask):,} ({np.sum(platform_mask)/len(points)*100:.1f}%)")
    print(f"STEEL STRUCT:   {np.sum(steel_mask):,} ({np.sum(steel_mask)/len(points)*100:.1f}%)")
    print(f"NOISE/OTHER:    {np.sum(noise_mask):,} ({np.sum(noise_mask)/len(points)*100:.1f}%)")

    # Generate visualizations
    print("\n--- Generating visualizations ---")
    plot_results(points, pipe_mask, platform_mask, steel_mask, noise_mask, output_dir)

    # Save LAS files
    print("\n--- Saving LAS files ---")
    save_las(str(output_dir / "pipes.las"), points[pipe_mask])
    save_las(str(output_dir / "platforms.las"), points[platform_mask])
    save_las(str(output_dir / "steel_structure.las"), points[steel_mask])
    save_las(str(output_dir / "noise.las"), points[noise_mask])

    print("\n" + "=" * 60)
    print("STEP 2 COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print("\nNext step: Use steel_structure.las for beam/bunton classification")


if __name__ == "__main__":
    main()
