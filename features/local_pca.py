"""
Local PCA feature computation for point-wise geometric descriptors.
"""

import numpy as np
from scipy.spatial import KDTree
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import K_NEIGHBORS, MIN_NEIGHBORS


def compute_local_pca(points: np.ndarray, k: int = None,
                      verbose: bool = True) -> dict:
    """
    Compute local PCA features for each point.

    For each point, we analyze its k-nearest neighborhood and extract:
    - Eigenvalues (λ1 >= λ2 >= λ3)
    - Linearity: (λ1 - λ2) / λ1 - high for linear structures (beams)
    - Planarity: (λ2 - λ3) / λ1 - high for planar structures (platforms)
    - Scattering: λ3 / λ1 - high for volumetric/noisy regions
    - Principal direction: eigenvector corresponding to λ1

    Args:
        points: Nx3 array of XYZ coordinates
        k: Number of neighbors (default from config)
        verbose: Print progress

    Returns:
        Dictionary with feature arrays:
        - linearity: N array
        - planarity: N array
        - scattering: N array
        - directions: Nx3 array of principal directions
        - eigenvalues: Nx3 array of [λ1, λ2, λ3]
        - normals: Nx3 array (eigenvector of smallest eigenvalue)
    """
    if k is None:
        k = K_NEIGHBORS

    n_points = len(points)

    # Build KD-tree for neighbor queries
    if verbose:
        print(f"Building KD-tree for {n_points:,} points...")
    tree = KDTree(points)

    # Initialize output arrays
    linearity = np.zeros(n_points)
    planarity = np.zeros(n_points)
    scattering = np.zeros(n_points)
    directions = np.zeros((n_points, 3))
    eigenvalues = np.zeros((n_points, 3))
    normals = np.zeros((n_points, 3))

    if verbose:
        print(f"Computing local PCA (k={k})...")

    # Query all neighbors at once (more efficient)
    distances, indices = tree.query(points, k=k + 1)  # +1 includes self

    # Process each point
    report_interval = n_points // 10
    for i in range(n_points):
        if verbose and i > 0 and i % report_interval == 0:
            print(f"  {i/n_points*100:.0f}% complete...")

        # Get neighbors (exclude self)
        neighbor_idx = indices[i, 1:]

        if len(neighbor_idx) < MIN_NEIGHBORS:
            # Not enough neighbors - mark as scattered
            scattering[i] = 1.0
            directions[i] = [0, 0, 1]  # Default vertical
            normals[i] = [0, 0, 1]
            continue

        # Get neighbor points
        neighbors = points[neighbor_idx]

        # Compute covariance matrix
        centered = neighbors - np.mean(neighbors, axis=0)
        cov = np.cov(centered.T)

        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort descending (eigh returns ascending)
        sort_idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sort_idx]
        eigvecs = eigvecs[:, sort_idx]

        # Ensure positive eigenvalues (numerical stability)
        eigvals = np.maximum(eigvals, 1e-10)

        # Store eigenvalues
        eigenvalues[i] = eigvals

        # Compute features
        lambda1, lambda2, lambda3 = eigvals

        linearity[i] = (lambda1 - lambda2) / lambda1
        planarity[i] = (lambda2 - lambda3) / lambda1
        scattering[i] = lambda3 / lambda1

        # Principal direction (eigenvector of largest eigenvalue)
        directions[i] = eigvecs[:, 0]

        # Normal (eigenvector of smallest eigenvalue) - for planar surfaces
        normals[i] = eigvecs[:, 2]

    if verbose:
        print("Local PCA complete.")
        print(f"  Linearity: min={linearity.min():.3f}, max={linearity.max():.3f}, mean={linearity.mean():.3f}")
        print(f"  Planarity: min={planarity.min():.3f}, max={planarity.max():.3f}, mean={planarity.mean():.3f}")
        print(f"  Scattering: min={scattering.min():.3f}, max={scattering.max():.3f}, mean={scattering.mean():.3f}")

    return {
        'linearity': linearity,
        'planarity': planarity,
        'scattering': scattering,
        'directions': directions,
        'eigenvalues': eigenvalues,
        'normals': normals
    }


def classify_by_geometry(features: dict, linearity_thresh: float = 0.7,
                         planarity_thresh: float = 0.6) -> np.ndarray:
    """
    Basic geometric classification based on PCA features.

    Args:
        features: Output from compute_local_pca
        linearity_thresh: Threshold for linear classification
        planarity_thresh: Threshold for planar classification

    Returns:
        geometry_type: N array with values:
            0 = unclassified
            1 = linear (beam-like)
            2 = planar (platform-like)
            3 = scattered (noise)
    """
    n = len(features['linearity'])
    geometry_type = np.zeros(n, dtype=np.int32)

    linear_mask = features['linearity'] > linearity_thresh
    planar_mask = (features['planarity'] > planarity_thresh) & ~linear_mask
    scatter_mask = features['scattering'] > 0.5

    geometry_type[linear_mask] = 1
    geometry_type[planar_mask] = 2
    geometry_type[scatter_mask] = 3

    print(f"Geometry classification:")
    print(f"  Linear: {np.sum(linear_mask):,} ({np.sum(linear_mask)/n*100:.1f}%)")
    print(f"  Planar: {np.sum(planar_mask):,} ({np.sum(planar_mask)/n*100:.1f}%)")
    print(f"  Scattered: {np.sum(scatter_mask):,} ({np.sum(scatter_mask)/n*100:.1f}%)")
    print(f"  Unclassified: {np.sum(geometry_type==0):,} ({np.sum(geometry_type==0)/n*100:.1f}%)")

    return geometry_type
