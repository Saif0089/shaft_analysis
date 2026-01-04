#!/usr/bin/env python3
"""
Step 2: Separate Pipes/Wires from Steel Structure

Input: output/step1_wall_stripped/interior.las
Output: output/step2_pipes_extracted/
    - pipes_and_wires.las (smooth, curved, cylindrical elements)
    - steel_structure.las (sharp, angular steel elements)

Key Distinction:
- PIPES/WIRES: Smooth, curved/arched surfaces, cylindrical cross-section
  They run vertically but have gentle curvature (not straight lines)
- STEEL STRUCTURE: Sharp edges, angular shapes (I-beams, angles, channels)
  Can be any orientation (vertical, horizontal, diagonal)

Detection Strategy:
1. Local PCA to get surface normals and principal directions
2. Compute local curvature variation - pipes have smooth consistent curvature
3. Analyze eigenvalue ratios - pipes are cylindrical (λ1 >> λ2 ≈ λ3)
4. Detect sharp edges via normal variation - steel has discontinuous normals
5. Cross-section analysis - pipes are circular, steel is rectangular/angular

The key is: pipes are SMOOTH, steel is SHARP/ANGULAR
"""

import numpy as np
import laspy
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from pathlib import Path


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


def compute_local_features(points, k=25):
    """
    Compute local geometric features for each point.

    For pipe/steel separation, we compute:
    - Eigenvalue ratios to detect cylindrical vs planar/linear shapes
    - Normal vectors for edge detection
    - Curvature estimates

    Returns dict with:
        - cylindricality: high for pipes (λ1 >> λ2 ≈ λ3)
        - linearity: high for beams (λ1 >> λ2 >> λ3)
        - planarity: high for flat surfaces
        - normals: surface normal vectors
        - curvature: local curvature estimate
    """
    n = len(points)
    print(f"Computing local features for {n:,} points (k={k})...")

    tree = KDTree(points)
    _, indices = tree.query(points, k=k+1)

    # Output arrays
    eigenvalues = np.zeros((n, 3))
    linearity = np.zeros(n)
    planarity = np.zeros(n)
    cylindricality = np.zeros(n)
    normals = np.zeros((n, 3))
    curvature = np.zeros(n)
    normal_variation = np.zeros(n)  # How much normals vary locally (high = edge)

    for i in range(n):
        if i % 50000 == 0 and i > 0:
            print(f"  {i/n*100:.0f}%...")

        # Get neighbor points (exclude self)
        neighbor_idx = indices[i, 1:]
        neighbors = points[neighbor_idx]

        # Compute covariance matrix
        centroid = np.mean(neighbors, axis=0)
        centered = neighbors - centroid
        cov = np.cov(centered.T)

        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort descending (λ1 >= λ2 >= λ3)
        idx = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[idx], 1e-10)
        eigvecs = eigvecs[:, idx]

        eigenvalues[i] = eigvals
        l1, l2, l3 = eigvals

        # Linearity: high when λ1 >> λ2 (stick-like, beams)
        linearity[i] = (l1 - l2) / l1

        # Planarity: high when λ2 >> λ3 (flat surface)
        planarity[i] = (l2 - l3) / l1

        # Cylindricality: ratio of λ2/λ3 when both are small compared to λ1
        # For a cylinder: λ1 is along axis, λ2 ≈ λ3 (circular cross-section)
        # For I-beam: λ2 >> λ3 (rectangular cross-section)
        if l2 > 1e-8 and l3 > 1e-8:
            cylindricality[i] = min(l2, l3) / max(l2, l3)  # Close to 1 for circular
        else:
            cylindricality[i] = 0

        # Surface normal (eigenvector of smallest eigenvalue)
        normals[i] = eigvecs[:, 2]

        # Curvature estimate (ratio of smallest to sum)
        curvature[i] = l3 / (l1 + l2 + l3)

    print("  Computing normal variation (edge detection)...")

    # Second pass: compute normal variation (how much normals differ from neighbors)
    # High variation = sharp edge, Low variation = smooth surface
    for i in range(n):
        if i % 100000 == 0 and i > 0:
            print(f"  {i/n*100:.0f}%...")

        neighbor_idx = indices[i, 1:k//2]  # Use fewer neighbors for edge sensitivity
        neighbor_normals = normals[neighbor_idx]

        # Compute angular deviation from mean normal
        my_normal = normals[i]

        # Dot products with neighbor normals (1 = same direction, 0 = perpendicular)
        dots = np.abs(np.sum(neighbor_normals * my_normal, axis=1))

        # Normal variation: low dots mean high variation (edges)
        normal_variation[i] = 1 - np.mean(dots)

    print("  Done.")

    return {
        'eigenvalues': eigenvalues,
        'linearity': linearity,
        'planarity': planarity,
        'cylindricality': cylindricality,
        'normals': normals,
        'curvature': curvature,
        'normal_variation': normal_variation
    }


def classify_pipe_vs_steel(points, features):
    """
    Classify points as pipe/wire vs steel structure.

    PIPES/WIRES characteristics:
    - Cylindrical cross-section: cylindricality close to 1
    - Smooth surface: low normal variation
    - Can be linear (straight pipe) or curved

    STEEL STRUCTURE characteristics:
    - Angular cross-section: cylindricality low (I-beams, angles)
    - Sharp edges: high normal variation at edges
    - Linear elements but with planar faces

    The key differentiator is:
    - Pipes: smooth, circular cross-section (λ2 ≈ λ3)
    - Steel: angular, rectangular cross-section (λ2 >> λ3 or sharp edges)
    """

    cylindricality = features['cylindricality']
    linearity = features['linearity']
    planarity = features['planarity']
    normal_variation = features['normal_variation']
    curvature = features['curvature']

    n = len(points)

    print("\nClassifying pipe vs steel...")
    print(f"  Cylindricality: min={cylindricality.min():.3f}, max={cylindricality.max():.3f}, mean={cylindricality.mean():.3f}")
    print(f"  Normal variation: min={normal_variation.min():.3f}, max={normal_variation.max():.3f}, mean={normal_variation.mean():.3f}")
    print(f"  Linearity: min={linearity.min():.3f}, max={linearity.max():.3f}, mean={linearity.mean():.3f}")

    # PIPE/WIRE criteria:
    # 1. High cylindricality (circular cross-section) OR
    # 2. Linear but smooth (low normal variation)

    # Cylindrical elements (round cross-section)
    is_cylindrical = cylindricality > 0.7

    # Smooth surfaces (not at sharp edges)
    is_smooth = normal_variation < 0.15

    # Linear elements
    is_linear = linearity > 0.5

    # PIPE = cylindrical OR (linear AND smooth)
    # Pipes can be straight (linear) or curved, but always smooth and round
    pipe_mask = is_cylindrical | (is_linear & is_smooth & (cylindricality > 0.5))

    # Additional: very high cylindricality even if some normal variation
    very_cylindrical = cylindricality > 0.85
    pipe_mask = pipe_mask | very_cylindrical

    # STEEL = everything else that is structural (not too scattered)
    # Steel has either: low cylindricality (angular) OR high normal variation (edges)
    is_structural = (linearity > 0.3) | (planarity > 0.3)  # Not scattered noise
    steel_mask = ~pipe_mask & is_structural

    # Anything remaining is noise/other (platforms, scattered points)
    noise_mask = ~pipe_mask & ~steel_mask

    print(f"\nInitial classification:")
    print(f"  Pipe candidates: {np.sum(pipe_mask):,} ({np.sum(pipe_mask)/n*100:.1f}%)")
    print(f"  Steel candidates: {np.sum(steel_mask):,} ({np.sum(steel_mask)/n*100:.1f}%)")
    print(f"  Noise/other: {np.sum(noise_mask):,} ({np.sum(noise_mask)/n*100:.1f}%)")

    return pipe_mask, steel_mask, noise_mask


def refine_classification_by_continuity(points, pipe_mask, steel_mask,
                                         radius=0.15, min_cluster_size=50):
    """
    Refine classification using spatial continuity.

    - Small isolated "pipe" clusters are likely noise or misclassified steel
    - Large continuous pipe regions are true pipes

    This helps clean up the boundaries between pipes and steel.
    """
    print("\nRefining classification by spatial continuity...")

    # Build KDTree on pipe points
    pipe_indices = np.where(pipe_mask)[0]
    if len(pipe_indices) < min_cluster_size:
        return pipe_mask, steel_mask

    pipe_points = points[pipe_indices]
    tree = KDTree(pipe_points)

    # Find connected components among pipe points
    visited = np.zeros(len(pipe_points), dtype=bool)
    clusters = []

    for start in range(len(pipe_points)):
        if visited[start]:
            continue

        # BFS to find cluster
        cluster = []
        queue = [start]

        while queue:
            curr = queue.pop(0)
            if visited[curr]:
                continue
            visited[curr] = True
            cluster.append(curr)

            neighbors = tree.query_ball_point(pipe_points[curr], radius)
            for n in neighbors:
                if not visited[n]:
                    queue.append(n)

        clusters.append(cluster)

    # Keep only large clusters as pipes
    refined_pipe_mask = np.zeros(len(points), dtype=bool)

    large_clusters = 0
    small_clusters = 0

    for cluster in clusters:
        if len(cluster) >= min_cluster_size:
            # Large cluster - keep as pipe
            original_indices = pipe_indices[cluster]
            refined_pipe_mask[original_indices] = True
            large_clusters += 1
        else:
            # Small cluster - reclassify as steel
            small_clusters += 1

    print(f"  Found {len(clusters)} clusters: {large_clusters} large (kept as pipe), {small_clusters} small (moved to steel)")

    # Update steel mask: original steel + small pipe clusters
    refined_steel_mask = steel_mask | (pipe_mask & ~refined_pipe_mask)

    return refined_pipe_mask, refined_steel_mask


def analyze_cross_section(points, features, pipe_mask, sample_z_levels=5):
    """
    Analyze cross-sections at various Z levels to validate pipe detection.
    This is for diagnostic purposes.
    """
    print("\nCross-section analysis at sample Z levels:")

    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    z_levels = np.linspace(z_min + 0.5, z_max - 0.5, sample_z_levels)

    for z in z_levels:
        # Get points near this Z level
        z_mask = np.abs(points[:, 2] - z) < 0.1

        n_total = np.sum(z_mask)
        n_pipe = np.sum(z_mask & pipe_mask)

        if n_total > 0:
            print(f"  Z={z:.1f}m: {n_pipe}/{n_total} pipe points ({n_pipe/n_total*100:.1f}%)")


def plot_results(points, pipe_mask, steel_mask, output_dir):
    """Generate visualization of results."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # XY view - Pipes
    ax = axes[0, 0]
    if np.sum(pipe_mask) > 0:
        ax.scatter(points[pipe_mask, 0], points[pipe_mask, 1],
                   s=0.1, alpha=0.5, c='red')
    ax.set_title(f'PIPES/WIRES (smooth, curved)\n{np.sum(pipe_mask):,} pts ({np.sum(pipe_mask)/len(points)*100:.1f}%)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # XY view - Steel
    ax = axes[0, 1]
    if np.sum(steel_mask) > 0:
        ax.scatter(points[steel_mask, 0], points[steel_mask, 1],
                   s=0.1, alpha=0.5, c='blue')
    ax.set_title(f'STEEL STRUCTURE (sharp, angular)\n{np.sum(steel_mask):,} pts ({np.sum(steel_mask)/len(points)*100:.1f}%)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # XZ view - Pipes
    ax = axes[1, 0]
    if np.sum(pipe_mask) > 0:
        ax.scatter(points[pipe_mask, 0], points[pipe_mask, 2],
                   s=0.1, alpha=0.5, c='red')
    ax.set_title('PIPES/WIRES - Side View')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.grid(True, alpha=0.3)

    # XZ view - Steel
    ax = axes[1, 1]
    if np.sum(steel_mask) > 0:
        ax.scatter(points[steel_mask, 0], points[steel_mask, 2],
                   s=0.1, alpha=0.5, c='blue')
    ax.set_title('STEEL STRUCTURE - Side View')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_dir / "step2_results.png"), dpi=150)
    plt.close()
    print(f"Saved: step2_results.png")

    # Combined view
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # XY combined
    ax = axes[0]
    if np.sum(steel_mask) > 0:
        ax.scatter(points[steel_mask, 0], points[steel_mask, 1],
                   s=0.1, alpha=0.3, c='blue', label=f'Steel ({np.sum(steel_mask):,})')
    if np.sum(pipe_mask) > 0:
        ax.scatter(points[pipe_mask, 0], points[pipe_mask, 1],
                   s=0.1, alpha=0.5, c='red', label=f'Pipes ({np.sum(pipe_mask):,})')
    ax.set_title('Step 2: Pipes vs Steel - Top View (XY)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.legend(markerscale=10)
    ax.grid(True, alpha=0.3)

    # XZ combined
    ax = axes[1]
    if np.sum(steel_mask) > 0:
        ax.scatter(points[steel_mask, 0], points[steel_mask, 2],
                   s=0.1, alpha=0.3, c='blue', label=f'Steel ({np.sum(steel_mask):,})')
    if np.sum(pipe_mask) > 0:
        ax.scatter(points[pipe_mask, 0], points[pipe_mask, 2],
                   s=0.1, alpha=0.5, c='red', label=f'Pipes ({np.sum(pipe_mask):,})')
    ax.set_title('Step 2: Pipes vs Steel - Side View (XZ)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.legend(markerscale=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_dir / "step2_combined.png"), dpi=150)
    plt.close()
    print(f"Saved: step2_combined.png")

    # Feature distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # These will be filled in main() with feature data
    plt.tight_layout()
    plt.savefig(str(output_dir / "step2_features.png"), dpi=150)
    plt.close()
    print(f"Saved: step2_features.png")


def plot_feature_distributions(features, pipe_mask, steel_mask, output_dir):
    """Plot feature distributions for pipe vs steel."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Cylindricality distribution
    ax = axes[0, 0]
    ax.hist(features['cylindricality'][pipe_mask], bins=50, alpha=0.7,
            label='Pipes', color='red', density=True)
    ax.hist(features['cylindricality'][steel_mask], bins=50, alpha=0.7,
            label='Steel', color='blue', density=True)
    ax.set_xlabel('Cylindricality')
    ax.set_ylabel('Density')
    ax.set_title('Cylindricality Distribution\n(High = circular cross-section)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Normal variation distribution
    ax = axes[0, 1]
    ax.hist(features['normal_variation'][pipe_mask], bins=50, alpha=0.7,
            label='Pipes', color='red', density=True)
    ax.hist(features['normal_variation'][steel_mask], bins=50, alpha=0.7,
            label='Steel', color='blue', density=True)
    ax.set_xlabel('Normal Variation')
    ax.set_ylabel('Density')
    ax.set_title('Normal Variation Distribution\n(High = sharp edges)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Linearity distribution
    ax = axes[1, 0]
    ax.hist(features['linearity'][pipe_mask], bins=50, alpha=0.7,
            label='Pipes', color='red', density=True)
    ax.hist(features['linearity'][steel_mask], bins=50, alpha=0.7,
            label='Steel', color='blue', density=True)
    ax.set_xlabel('Linearity')
    ax.set_ylabel('Density')
    ax.set_title('Linearity Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scatter: cylindricality vs normal variation
    ax = axes[1, 1]
    sample_size = min(10000, np.sum(pipe_mask), np.sum(steel_mask))
    if sample_size > 0:
        pipe_idx = np.random.choice(np.where(pipe_mask)[0], min(sample_size, np.sum(pipe_mask)), replace=False)
        steel_idx = np.random.choice(np.where(steel_mask)[0], min(sample_size, np.sum(steel_mask)), replace=False)

        ax.scatter(features['cylindricality'][steel_idx], features['normal_variation'][steel_idx],
                   s=1, alpha=0.3, c='blue', label='Steel')
        ax.scatter(features['cylindricality'][pipe_idx], features['normal_variation'][pipe_idx],
                   s=1, alpha=0.3, c='red', label='Pipes')
    ax.set_xlabel('Cylindricality')
    ax.set_ylabel('Normal Variation')
    ax.set_title('Feature Space\n(Pipes: high cyl, low var | Steel: low cyl, high var)')
    ax.legend(markerscale=5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_dir / "step2_features.png"), dpi=150)
    plt.close()
    print(f"Saved: step2_features.png")


def main():
    # =========================================================================
    # Configuration
    # =========================================================================
    input_path = Path("output/step1_wall_stripped/interior.las")
    output_dir = Path("output/step2_pipes_extracted")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 2: SEPARATE PIPES/WIRES FROM STEEL STRUCTURE")
    print("=" * 70)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_dir}")
    print("\nKey distinction:")
    print("  PIPES/WIRES: Smooth, curved, circular cross-section")
    print("  STEEL:       Sharp, angular, rectangular cross-section")

    # =========================================================================
    # Step 1: Load interior points from step 1
    # =========================================================================
    print("\n" + "-" * 70)
    print("[1/5] Loading interior points from Step 1...")
    print("-" * 70)
    points = load_las(str(input_path))

    # =========================================================================
    # Step 2: Compute local geometric features
    # =========================================================================
    print("\n" + "-" * 70)
    print("[2/5] Computing local geometric features...")
    print("-" * 70)
    print("  - Eigenvalues for shape analysis")
    print("  - Cylindricality (λ2/λ3 ratio - high for circular cross-section)")
    print("  - Normal variation (high at sharp edges)")

    features = compute_local_features(points, k=25)

    # =========================================================================
    # Step 3: Classify pipe vs steel
    # =========================================================================
    print("\n" + "-" * 70)
    print("[3/5] Classifying points as pipe/wire vs steel...")
    print("-" * 70)

    pipe_mask, steel_mask, noise_mask = classify_pipe_vs_steel(points, features)

    # =========================================================================
    # Step 4: Refine by spatial continuity
    # =========================================================================
    print("\n" + "-" * 70)
    print("[4/5] Refining classification by spatial continuity...")
    print("-" * 70)

    pipe_mask, steel_mask = refine_classification_by_continuity(
        points, pipe_mask, steel_mask,
        radius=0.12,
        min_cluster_size=100
    )

    # Add noise back to steel (we don't output noise separately)
    steel_mask = steel_mask | noise_mask

    # Cross-section analysis for validation
    analyze_cross_section(points, features, pipe_mask)

    # =========================================================================
    # Step 5: Save results
    # =========================================================================
    print("\n" + "-" * 70)
    print("[5/5] Saving results...")
    print("-" * 70)

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL CLASSIFICATION")
    print(f"{'='*70}")
    print(f"PIPES/WIRES:     {np.sum(pipe_mask):,} points ({np.sum(pipe_mask)/len(points)*100:.1f}%)")
    print(f"STEEL STRUCTURE: {np.sum(steel_mask):,} points ({np.sum(steel_mask)/len(points)*100:.1f}%)")

    # Generate visualizations
    print("\n--- Generating visualizations ---")
    plot_results(points, pipe_mask, steel_mask, output_dir)
    plot_feature_distributions(features, pipe_mask, steel_mask, output_dir)

    # Save LAS files
    print("\n--- Saving LAS files ---")
    save_las(str(output_dir / "pipes_and_wires.las"), points[pipe_mask])
    save_las(str(output_dir / "steel_structure.las"), points[steel_mask])

    print(f"\n{'='*70}")
    print("STEP 2 COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  - pipes_and_wires.las  : Smooth curved elements")
    print(f"  - steel_structure.las  : Angular steel elements")
    print(f"\nNext step: steel_structure.las ready for beam/bunton classification")


if __name__ == "__main__":
    main()
