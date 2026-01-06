"""
3D Point Cloud Annotation Tool
- Load LAS/PLY/PCD point clouds
- Create 3D bounding boxes interactively
- Label bounding boxes
- Export annotations for PointNet++ training
"""

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import laspy
import json
import os
from pathlib import Path
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ANNOTATIONS_FOLDER'] = 'annotations'

# Create folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ANNOTATIONS_FOLDER'], exist_ok=True)

# Global state
current_data = {
    'points': None,
    'colors': None,
    'filename': None,
    'annotations': []
}

def load_las_file(filepath):
    """Load LAS point cloud file"""
    las = laspy.read(filepath)
    points = np.vstack([las.x, las.y, las.z]).T

    # Normalize to origin for easier viewing
    centroid = points.mean(axis=0)
    points = points - centroid

    # Get colors if available
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        colors = np.vstack([las.red, las.green, las.blue]).T / 65535.0
    else:
        # Use intensity as grayscale
        if hasattr(las, 'intensity'):
            intensity = las.intensity / las.intensity.max()
            colors = np.column_stack([intensity, intensity, intensity])
        else:
            colors = np.ones((len(points), 3)) * 0.5

    return points, colors, centroid

def load_ply_file(filepath):
    """Load PLY point cloud file"""
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(filepath)
    points = np.asarray(pcd.points)
    centroid = points.mean(axis=0)
    points = points - centroid

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.ones((len(points), 3)) * 0.5

    return points, colors, centroid

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save file
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Load point cloud
    ext = Path(filename).suffix.lower()
    try:
        if ext == '.las' or ext == '.laz':
            points, colors, centroid = load_las_file(filepath)
        elif ext == '.ply':
            points, colors, centroid = load_ply_file(filepath)
        else:
            return jsonify({'error': f'Unsupported format: {ext}'}), 400

        # Store ALL points (no subsampling - needed for accurate labeling)
        current_data['full_points'] = points.copy()
        current_data['full_colors'] = colors.copy()

        # Keep all points for display too (modern browsers can handle it)
        current_data['points'] = points
        current_data['colors'] = colors
        current_data['filename'] = filename
        current_data['centroid'] = centroid.tolist()
        current_data['annotations'] = []

        # Load existing annotations if any
        ann_file = os.path.join(app.config['ANNOTATIONS_FOLDER'], f"{Path(filename).stem}.json")
        if os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                current_data['annotations'] = json.load(f)['annotations']

        return jsonify({
            'success': True,
            'num_points': len(points),
            'filename': filename,
            'bounds': {
                'min': points.min(axis=0).tolist(),
                'max': points.max(axis=0).tolist()
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_points')
def get_points():
    if current_data['points'] is None:
        return jsonify({'error': 'No point cloud loaded'}), 400

    # Subsample for display (browser can't handle 1.4M points in JSON)
    # Full points are kept for export
    points = current_data['points']
    colors = current_data['colors']

    max_display = 300000  # 300K is reasonable for WebGL display
    if len(points) > max_display:
        indices = np.random.choice(len(points), max_display, replace=False)
        indices.sort()  # Keep spatial order
        display_points = points[indices]
        display_colors = colors[indices]
    else:
        display_points = points
        display_colors = colors

    return jsonify({
        'points': display_points.tolist(),
        'colors': display_colors.tolist(),
        'annotations': current_data['annotations'],
        'total_points': len(current_data.get('full_points', points)),
        'display_points': len(display_points)
    })

@app.route('/add_annotation', methods=['POST'])
def add_annotation():
    data = request.json

    annotation = {
        'id': str(uuid.uuid4())[:8],
        'label': data.get('label', 'unlabeled'),
        'bbox': {
            'min': data['bbox']['min'],
            'max': data['bbox']['max']
        },
        'color': data.get('color', '#ff0000')
    }

    current_data['annotations'].append(annotation)
    save_annotations()

    return jsonify({'success': True, 'annotation': annotation})

@app.route('/update_annotation', methods=['POST'])
def update_annotation():
    data = request.json
    ann_id = data['id']

    for ann in current_data['annotations']:
        if ann['id'] == ann_id:
            if 'label' in data:
                ann['label'] = data['label']
            if 'bbox' in data:
                ann['bbox'] = data['bbox']
            if 'color' in data:
                ann['color'] = data['color']
            break

    save_annotations()
    return jsonify({'success': True})

@app.route('/delete_annotation', methods=['POST'])
def delete_annotation():
    data = request.json
    ann_id = data['id']

    current_data['annotations'] = [a for a in current_data['annotations'] if a['id'] != ann_id]
    save_annotations()

    return jsonify({'success': True})

@app.route('/get_annotations')
def get_annotations():
    return jsonify({'annotations': current_data['annotations']})

@app.route('/save_all_annotations', methods=['POST'])
def save_all_annotations():
    """Save all annotations at once (from frontend)"""
    data = request.json
    current_data['annotations'] = data.get('annotations', [])
    save_annotations()
    return jsonify({'success': True})

def save_annotations():
    if current_data['filename']:
        ann_file = os.path.join(
            app.config['ANNOTATIONS_FOLDER'],
            f"{Path(current_data['filename']).stem}.json"
        )
        with open(ann_file, 'w') as f:
            json.dump({
                'filename': current_data['filename'],
                'centroid': current_data.get('centroid', [0, 0, 0]),
                'annotations': current_data['annotations']
            }, f, indent=2)

def rotation_matrix_from_euler(rotation_deg):
    """Create rotation matrix from Euler angles (degrees) - XYZ order"""
    rx, ry, rz = np.radians(rotation_deg)

    # Rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    # Combined rotation: R = Rz * Ry * Rx (same as Three.js default XYZ order)
    return Rz @ Ry @ Rx

def points_in_oriented_bbox(points, center, half_extents, rotation_matrix):
    """
    Check if points are inside an oriented bounding box.

    Args:
        points: Nx3 array of points
        center: 3D center of the box
        half_extents: half-size in each local axis (before rotation)
        rotation_matrix: 3x3 rotation matrix

    Returns:
        Boolean mask of points inside the OBB
    """
    # Transform points to box's local coordinate system
    # 1. Translate to box center
    local_points = points - center

    # 2. Inverse rotate (transpose of rotation matrix)
    local_points = local_points @ rotation_matrix  # R^T = R^-1 for rotation matrices

    # 3. Check if within half extents (axis-aligned check in local space)
    inside = np.all(np.abs(local_points) <= half_extents, axis=1)

    return inside

@app.route('/export_pointnet', methods=['GET'])
def export_pointnet():
    """Export annotations in PointNet++ training format"""
    from datetime import datetime

    if current_data['points'] is None:
        return jsonify({'error': 'No point cloud loaded'}), 400

    # Use FULL points for export (not subsampled display points)
    points = current_data.get('full_points', current_data['points'])
    colors = current_data.get('full_colors', current_data['colors'])
    annotations = current_data['annotations']

    # Create label array (-1 = background)
    labels = np.full(len(points), -1, dtype=np.int32)

    # Get unique labels
    label_names = list(set(ann['label'] for ann in annotations))
    label_to_id = {name: i for i, name in enumerate(label_names)}

    # Assign labels based on bounding boxes (with rotation support)
    for ann in annotations:
        bbox_min = np.array(ann['bbox']['min'])
        bbox_max = np.array(ann['bbox']['max'])

        # Calculate center and half extents
        center = (bbox_min + bbox_max) / 2
        half_extents = (bbox_max - bbox_min) / 2

        # Check if rotation is specified
        rotation = ann.get('rotation', [0, 0, 0])

        if any(r != 0 for r in rotation):
            # Use oriented bounding box check for rotated boxes
            rot_matrix = rotation_matrix_from_euler(rotation)
            inside = points_in_oriented_bbox(points, center, half_extents, rot_matrix)
        else:
            # Use fast axis-aligned check for non-rotated boxes
            inside = np.all((points >= bbox_min) & (points <= bbox_max), axis=1)

        labels[inside] = label_to_id[ann['label']]

    # Save in PointNet++ format with timestamp to preserve each export
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = Path(current_data['filename']).stem
    export_name = f'export_{stem}_{timestamp}'
    output_dir = os.path.join(app.config['ANNOTATIONS_FOLDER'], 'pointnet_format', export_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save points (x, y, z, r, g, b)
    point_data = np.column_stack([points, colors * 255])
    np.savetxt(os.path.join(output_dir, f'{stem}_points.txt'), point_data, fmt='%.6f')

    # Save labels
    np.savetxt(os.path.join(output_dir, f'{stem}_labels.txt'), labels, fmt='%d')

    # Save label mapping
    with open(os.path.join(output_dir, f'{stem}_label_map.json'), 'w') as f:
        json.dump({
            'label_to_id': label_to_id,
            'id_to_label': {v: k for k, v in label_to_id.items()}
        }, f, indent=2)

    # Also save bounding boxes for detection tasks
    bbox_data = []
    for ann in annotations:
        bbox_data.append({
            'label': ann['label'],
            'label_id': label_to_id[ann['label']],
            'bbox_min': ann['bbox']['min'],
            'bbox_max': ann['bbox']['max'],
            'center': [
                (ann['bbox']['min'][i] + ann['bbox']['max'][i]) / 2
                for i in range(3)
            ],
            'size': [
                ann['bbox']['max'][i] - ann['bbox']['min'][i]
                for i in range(3)
            ],
            'rotation': ann.get('rotation', [0, 0, 0])  # Include rotation for OBB
        })

    with open(os.path.join(output_dir, f'{stem}_bboxes.json'), 'w') as f:
        json.dump(bbox_data, f, indent=2)

    return jsonify({
        'success': True,
        'output_dir': output_dir,
        'export_name': export_name,
        'files': [
            f'{stem}_points.txt',
            f'{stem}_labels.txt',
            f'{stem}_label_map.json',
            f'{stem}_bboxes.json'
        ],
        'stats': {
            'total_points': len(points),
            'labeled_points': int(np.sum(labels >= 0)),
            'num_classes': len(label_names),
            'classes': label_names
        }
    })

@app.route('/list_files')
def list_files():
    """List available point cloud files"""
    files = []
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        if f.endswith(('.las', '.laz', '.ply')):
            files.append(f)

    # Also check parent directory for LAS files
    parent_dir = Path(app.config['UPLOAD_FOLDER']).parent.parent
    for f in parent_dir.glob('*.las'):
        files.append(str(f))

    return jsonify({'files': files})

@app.route('/load_file', methods=['POST'])
def load_file():
    """Load a specific file by path"""
    data = request.json
    filepath = data.get('filepath')

    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 400

    ext = Path(filepath).suffix.lower()
    try:
        if ext == '.las' or ext == '.laz':
            points, colors, centroid = load_las_file(filepath)
        elif ext == '.ply':
            points, colors, centroid = load_ply_file(filepath)
        else:
            return jsonify({'error': f'Unsupported format: {ext}'}), 400

        # Store ALL points (no subsampling - needed for accurate labeling)
        current_data['full_points'] = points.copy()
        current_data['full_colors'] = colors.copy()

        # Keep all points for display too
        filename = Path(filepath).name
        current_data['points'] = points
        current_data['colors'] = colors
        current_data['filename'] = filename
        current_data['centroid'] = centroid.tolist()
        current_data['annotations'] = []

        # Load existing annotations
        ann_file = os.path.join(app.config['ANNOTATIONS_FOLDER'], f"{Path(filename).stem}.json")
        if os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                current_data['annotations'] = json.load(f)['annotations']

        return jsonify({
            'success': True,
            'num_points': len(points),
            'filename': filename,
            'bounds': {
                'min': points.min(axis=0).tolist(),
                'max': points.max(axis=0).tolist()
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/list_exports')
def list_exports():
    """List all previously exported annotation sets"""
    exports = []
    export_dir = os.path.join(app.config['ANNOTATIONS_FOLDER'], 'pointnet_format')

    if os.path.exists(export_dir):
        for folder in sorted(os.listdir(export_dir), reverse=True):
            folder_path = os.path.join(export_dir, folder)
            if os.path.isdir(folder_path):
                # Find bbox file
                bbox_files = [f for f in os.listdir(folder_path) if f.endswith('_bboxes.json')]
                if bbox_files:
                    bbox_path = os.path.join(folder_path, bbox_files[0])
                    with open(bbox_path, 'r') as f:
                        bboxes = json.load(f)

                    # Get label map
                    label_map_files = [f for f in os.listdir(folder_path) if f.endswith('_label_map.json')]
                    labels = []
                    if label_map_files:
                        with open(os.path.join(folder_path, label_map_files[0]), 'r') as f:
                            label_data = json.load(f)
                            labels = list(label_data.get('label_to_id', {}).keys())

                    exports.append({
                        'name': folder,
                        'path': folder_path,
                        'num_boxes': len(bboxes),
                        'labels': labels
                    })

    return jsonify({'exports': exports})

@app.route('/load_export', methods=['POST'])
def load_export():
    """Load bounding boxes from a previous export"""
    data = request.json
    export_path = data.get('path')

    if not export_path or not os.path.exists(export_path):
        return jsonify({'error': 'Export not found'}), 400

    # Find bbox file
    bbox_files = [f for f in os.listdir(export_path) if f.endswith('_bboxes.json')]
    if not bbox_files:
        return jsonify({'error': 'No bounding boxes found in export'}), 400

    bbox_path = os.path.join(export_path, bbox_files[0])
    with open(bbox_path, 'r') as f:
        bboxes = json.load(f)

    # Convert to annotation format
    annotations = []
    colors = {
        'column': '#3366cc',
        'guard2': '#ff9900',
        'guard': '#33cc33',
        'wall': '#cc3333',
        'wire': '#990099',
        'column2': '#00cccc',
        'pipe': '#cccc33',
    }

    for bbox in bboxes:
        ann = {
            'id': str(uuid.uuid4())[:8],
            'label': bbox['label'],
            'bbox': {
                'min': bbox['bbox_min'],
                'max': bbox['bbox_max']
            },
            'color': colors.get(bbox['label'], '#ff0000'),
            'rotation': bbox.get('rotation', [0, 0, 0])  # Load rotation if available
        }
        annotations.append(ann)

    current_data['annotations'] = annotations
    save_annotations()

    return jsonify({
        'success': True,
        'num_boxes': len(annotations),
        'annotations': annotations
    })

if __name__ == '__main__':
    print("=" * 60)
    print("3D Point Cloud Annotation Tool")
    print("=" * 60)
    print("Open http://localhost:5000 in your browser")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
