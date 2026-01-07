"""Utilities for converting bounding box annotations to point labels."""
import numpy as np
from typing import List, Dict, Any
from scipy.spatial.transform import Rotation


def point_in_oriented_box(points: np.ndarray, box: Dict[str, Any]) -> np.ndarray:
    """
    Check which points are inside an oriented bounding box.

    Args:
        points: Nx3 array of point coordinates
        box: Dictionary with 'position', 'rotation', 'scale' keys
            - position: [x, y, z] center of box
            - rotation: [rx, ry, rz] Euler angles in radians
            - scale: [sx, sy, sz] half-extents of box

    Returns:
        Boolean array of length N, True if point is inside box
    """
    # Get box parameters
    center = np.array(box['position'])
    rotation = np.array(box['rotation'])
    half_extents = np.array(box['scale']) / 2  # scale is full size, need half

    # Create rotation matrix (inverse to transform points to box space)
    rot = Rotation.from_euler('xyz', rotation)
    rot_matrix_inv = rot.inv().as_matrix()

    # Transform points to box-local coordinates
    points_local = points - center
    points_local = points_local @ rot_matrix_inv.T

    # Check if points are within box bounds
    inside = np.all(np.abs(points_local) <= half_extents, axis=1)

    return inside


def boxes_to_labels(points: np.ndarray, boxes: List[Dict], classes: List[Dict]) -> np.ndarray:
    """
    Convert bounding box annotations to per-point labels.

    Args:
        points: Nx3 array of point coordinates
        boxes: List of box dictionaries with 'label', 'position', 'rotation', 'scale'
        classes: List of class dictionaries with 'id', 'name'

    Returns:
        Array of N label IDs (0 = unlabeled)
    """
    n_points = len(points)
    labels = np.zeros(n_points, dtype=np.int32)  # Default to unlabeled (0)

    # Create name to ID mapping
    name_to_id = {cls['name']: cls['id'] for cls in classes}

    # Process each box (later boxes override earlier ones for overlapping points)
    for box in boxes:
        label_name = box.get('label', 'unlabeled')
        label_id = name_to_id.get(label_name, 0)

        # Find points inside this box
        inside = point_in_oriented_box(points, box)

        # Assign label
        labels[inside] = label_id

    return labels


def labels_to_colors(labels: np.ndarray, classes: List[Dict]) -> np.ndarray:
    """
    Convert label IDs to RGB colors for visualization.

    Args:
        labels: Array of N label IDs
        classes: List of class dictionaries with 'id', 'color'

    Returns:
        Nx3 array of RGB colors (0-255)
    """
    n_points = len(labels)
    colors = np.ones((n_points, 3), dtype=np.uint8) * 128  # Default gray

    # Create ID to color mapping
    id_to_color = {}
    for cls in classes:
        # Parse hex color
        hex_color = cls['color'].lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        id_to_color[cls['id']] = (r, g, b)

    # Assign colors
    for label_id, color in id_to_color.items():
        mask = labels == label_id
        colors[mask] = color

    return colors


if __name__ == '__main__':
    # Test the functions
    np.random.seed(42)

    # Create test points (10x10x10 cube centered at origin)
    points = np.random.uniform(-5, 5, (10000, 3))

    # Create test boxes
    boxes = [
        {
            'id': 'box1',
            'label': 'class_a',
            'position': [0, 0, 0],
            'rotation': [0, 0, 0],
            'scale': [4, 4, 4]  # 4x4x4 box at center
        },
        {
            'id': 'box2',
            'label': 'class_b',
            'position': [2, 0, 0],
            'rotation': [0, 0, 0.5],  # Rotated box
            'scale': [2, 2, 2]
        }
    ]

    # Create test classes
    classes = [
        {'id': 0, 'name': 'unlabeled', 'color': '#808080'},
        {'id': 1, 'name': 'class_a', 'color': '#FF0000'},
        {'id': 2, 'name': 'class_b', 'color': '#00FF00'}
    ]

    # Test boxes_to_labels
    labels = boxes_to_labels(points, boxes, classes)
    unique, counts = np.unique(labels, return_counts=True)
    print("Label distribution:")
    for u, c in zip(unique, counts):
        class_name = classes[u]['name']
        print(f"  {class_name}: {c} points ({100*c/len(labels):.1f}%)")

    # Test labels_to_colors
    colors = labels_to_colors(labels, classes)
    print(f"Colors shape: {colors.shape}")
    print(f"Sample colors: {colors[:3]}")

    print("All tests passed!")
