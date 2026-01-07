"""LAS file reading and writing utilities."""
import numpy as np
import laspy
from typing import Tuple, Optional, Dict, Any
import os


def read_las(filepath: str) -> Dict[str, np.ndarray]:
    """
    Read a LAS file and return point cloud data.

    Args:
        filepath: Path to LAS file

    Returns:
        Dictionary with keys: 'xyz', 'rgb', 'classification' (if available)
    """
    las = laspy.read(filepath)

    # Get XYZ coordinates
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)

    # Get RGB if available
    rgb = None
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        # LAS RGB is typically 16-bit, normalize to 0-255
        red = np.array(las.red)
        green = np.array(las.green)
        blue = np.array(las.blue)

        # Check if 16-bit (values > 255)
        if red.max() > 255 or green.max() > 255 or blue.max() > 255:
            red = (red / 256).astype(np.uint8)
            green = (green / 256).astype(np.uint8)
            blue = (blue / 256).astype(np.uint8)
        else:
            red = red.astype(np.uint8)
            green = green.astype(np.uint8)
            blue = blue.astype(np.uint8)

        rgb = np.vstack([red, green, blue]).T

    # Get classification if available
    classification = None
    if hasattr(las, 'classification'):
        classification = np.array(las.classification).astype(np.int32)

    result = {
        'xyz': xyz,
        'rgb': rgb,
        'classification': classification,
        'num_points': len(xyz)
    }

    return result


def write_las(filepath: str, xyz: np.ndarray, rgb: Optional[np.ndarray] = None,
              classification: Optional[np.ndarray] = None) -> None:
    """
    Write point cloud data to a LAS file.

    Args:
        filepath: Output path
        xyz: Nx3 array of XYZ coordinates
        rgb: Optional Nx3 array of RGB values (0-255)
        classification: Optional N array of classification labels
    """
    # Create LAS header
    header = laspy.LasHeader(point_format=2, version="1.2")

    # Set scale and offset based on data
    xyz = np.asarray(xyz, dtype=np.float64)
    header.scale = [0.001, 0.001, 0.001]  # 1mm precision
    header.offset = [
        np.floor(xyz[:, 0].min()),
        np.floor(xyz[:, 1].min()),
        np.floor(xyz[:, 2].min())
    ]

    # Create LAS data
    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]

    # Add RGB if provided
    if rgb is not None:
        rgb = np.asarray(rgb, dtype=np.uint16)
        # Convert 8-bit to 16-bit for LAS
        if rgb.max() <= 255:
            rgb = rgb * 256
        las.red = rgb[:, 0]
        las.green = rgb[:, 1]
        las.blue = rgb[:, 2]

    # Add classification if provided
    if classification is not None:
        las.classification = np.asarray(classification, dtype=np.uint8)

    # Write file
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    las.write(filepath)


def get_las_bounds(filepath: str) -> Dict[str, Tuple[float, float]]:
    """
    Get the bounding box of a LAS file without loading all points.

    Args:
        filepath: Path to LAS file

    Returns:
        Dictionary with 'x', 'y', 'z' keys, each containing (min, max) tuple
    """
    with laspy.open(filepath) as f:
        header = f.header
        return {
            'x': (header.x_min, header.x_max),
            'y': (header.y_min, header.y_max),
            'z': (header.z_min, header.z_max)
        }


def subsample_points(xyz: np.ndarray, rgb: Optional[np.ndarray] = None,
                     classification: Optional[np.ndarray] = None,
                     max_points: int = 500000) -> Tuple[np.ndarray, ...]:
    """
    Subsample point cloud to max_points using random sampling.

    Args:
        xyz: Nx3 array
        rgb: Optional Nx3 array
        classification: Optional N array
        max_points: Maximum number of points to return

    Returns:
        Tuple of subsampled arrays (xyz, rgb, classification) - None arrays stay None
    """
    n_points = len(xyz)
    if n_points <= max_points:
        return xyz, rgb, classification

    # Random subsample
    indices = np.random.choice(n_points, max_points, replace=False)
    indices = np.sort(indices)

    xyz_sub = xyz[indices]
    rgb_sub = rgb[indices] if rgb is not None else None
    cls_sub = classification[indices] if classification is not None else None

    return xyz_sub, rgb_sub, cls_sub


if __name__ == '__main__':
    # Test the functions
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import SLICES_DIR

    test_file = os.path.join(SLICES_DIR, 'slice_053.las')
    if os.path.exists(test_file):
        print(f"Testing with {test_file}")

        # Test read
        data = read_las(test_file)
        print(f"Points: {data['num_points']:,}")
        print(f"XYZ shape: {data['xyz'].shape}")
        print(f"RGB shape: {data['rgb'].shape if data['rgb'] is not None else 'None'}")
        print(f"X range: {data['xyz'][:, 0].min():.2f} to {data['xyz'][:, 0].max():.2f}")
        print(f"Y range: {data['xyz'][:, 1].min():.2f} to {data['xyz'][:, 1].max():.2f}")
        print(f"Z range: {data['xyz'][:, 2].min():.2f} to {data['xyz'][:, 2].max():.2f}")

        # Test bounds
        bounds = get_las_bounds(test_file)
        print(f"Bounds: {bounds}")

        # Test subsample
        xyz_sub, rgb_sub, _ = subsample_points(data['xyz'], data['rgb'], max_points=100000)
        print(f"Subsampled to {len(xyz_sub):,} points")

        print("All tests passed!")
    else:
        print(f"Test file not found: {test_file}")
