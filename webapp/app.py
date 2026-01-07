"""Flask backend for shaft segmentation annotation tool."""
import os
import sys
import json
import glob
from datetime import datetime
from flask import Flask, render_template, jsonify, request, send_from_directory

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SLICES_DIR, ANNOTATIONS_DIR, GROUND_TRUTH_DIR, MODELS_DIR,
    WEBAPP_HOST, WEBAPP_PORT, MAX_POINTS_DISPLAY, SOURCE_SLICES_DIR
)
from utils.las_io import read_las, write_las, subsample_points, get_las_bounds

app = Flask(__name__)

# Labels file path
LABELS_FILE = os.path.join(ANNOTATIONS_DIR, 'labels.json')


def get_labels():
    """Load labels from file or return defaults."""
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r') as f:
            return json.load(f)
    return {
        'classes': [
            {'id': 0, 'name': 'unlabeled', 'color': '#808080'}
        ]
    }


def save_labels(labels):
    """Save labels to file."""
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    with open(LABELS_FILE, 'w') as f:
        json.dump(labels, f, indent=2)


# ============== Routes ==============

@app.route('/')
def index():
    """Serve the main annotation interface."""
    return render_template('index.html')


@app.route('/api/slices')
def list_slices():
    """List all available slices."""
    slices = []

    # Get slices from working directory
    local_files = glob.glob(os.path.join(SLICES_DIR, '*.las'))

    for f in sorted(local_files):
        name = os.path.basename(f).replace('.las', '')
        bounds = get_las_bounds(f)
        slices.append({
            'name': name,
            'path': f,
            'z_min': bounds['z'][0],
            'z_max': bounds['z'][1],
            'has_annotation': os.path.exists(os.path.join(ANNOTATIONS_DIR, f'{name}.json')),
            'has_ground_truth': os.path.exists(os.path.join(GROUND_TRUTH_DIR, f'{name}_gt.las')),
            'has_prediction': os.path.exists(os.path.join(GROUND_TRUTH_DIR, f'{name}_pred.las'))
        })

    return jsonify({'slices': slices})


@app.route('/api/available-slices')
def list_available_slices():
    """List all slices available from source (for copying)."""
    slices = []
    source_files = glob.glob(os.path.join(SOURCE_SLICES_DIR, '*.las'))

    for f in sorted(source_files):
        name = os.path.basename(f).replace('.las', '')
        # Check if already in working directory
        local_exists = os.path.exists(os.path.join(SLICES_DIR, f'{name}.las'))
        slices.append({
            'name': name,
            'local_exists': local_exists
        })

    return jsonify({'slices': slices})


@app.route('/api/copy-slice/<name>', methods=['POST'])
def copy_slice(name):
    """Copy a slice from source to working directory."""
    source_path = os.path.join(SOURCE_SLICES_DIR, f'{name}.las')
    dest_path = os.path.join(SLICES_DIR, f'{name}.las')

    if not os.path.exists(source_path):
        return jsonify({'error': f'Source slice not found: {name}'}), 404

    if os.path.exists(dest_path):
        return jsonify({'message': 'Slice already exists locally', 'path': dest_path})

    # Copy file
    import shutil
    shutil.copy2(source_path, dest_path)

    return jsonify({'message': 'Slice copied successfully', 'path': dest_path})


@app.route('/api/slice/<name>')
def get_slice(name):
    """Get point cloud data for a slice."""
    filepath = os.path.join(SLICES_DIR, f'{name}.las')

    if not os.path.exists(filepath):
        return jsonify({'error': f'Slice not found: {name}'}), 404

    # Read LAS file
    data = read_las(filepath)

    # Subsample if too many points
    xyz, rgb, classification = subsample_points(
        data['xyz'], data['rgb'], data['classification'],
        max_points=MAX_POINTS_DISPLAY
    )

    # Prepare response - convert to lists for JSON
    response = {
        'name': name,
        'num_points': len(xyz),
        'total_points': data['num_points'],
        'x': xyz[:, 0].tolist(),
        'y': xyz[:, 1].tolist(),
        'z': xyz[:, 2].tolist(),
    }

    # Add RGB if available
    if rgb is not None:
        response['r'] = rgb[:, 0].tolist()
        response['g'] = rgb[:, 1].tolist()
        response['b'] = rgb[:, 2].tolist()

    # Add classification if available
    if classification is not None:
        response['classification'] = classification.tolist()

    return jsonify(response)


@app.route('/api/labels', methods=['GET'])
def get_labels_endpoint():
    """Get current label classes."""
    return jsonify(get_labels())


@app.route('/api/labels', methods=['POST'])
def update_labels():
    """Add or update label classes."""
    data = request.json

    if 'action' not in data:
        return jsonify({'error': 'Action required'}), 400

    labels = get_labels()

    if data['action'] == 'add':
        # Add new class
        name = data.get('name', '').strip()
        color = data.get('color', '#FF0000')

        if not name:
            return jsonify({'error': 'Name required'}), 400

        # Check if name already exists
        for cls in labels['classes']:
            if cls['name'].lower() == name.lower():
                return jsonify({'error': 'Class name already exists'}), 400

        # Get next ID
        max_id = max(cls['id'] for cls in labels['classes'])
        new_id = max_id + 1

        labels['classes'].append({
            'id': new_id,
            'name': name,
            'color': color
        })

    elif data['action'] == 'remove':
        class_id = data.get('id')
        if class_id is None:
            return jsonify({'error': 'Class ID required'}), 400

        if class_id == 0:
            return jsonify({'error': 'Cannot remove unlabeled class'}), 400

        labels['classes'] = [c for c in labels['classes'] if c['id'] != class_id]

    elif data['action'] == 'update':
        class_id = data.get('id')
        if class_id is None:
            return jsonify({'error': 'Class ID required'}), 400

        for cls in labels['classes']:
            if cls['id'] == class_id:
                if 'name' in data:
                    cls['name'] = data['name']
                if 'color' in data:
                    cls['color'] = data['color']
                break

    save_labels(labels)
    return jsonify(labels)


@app.route('/api/annotations/<name>', methods=['GET'])
def get_annotations(name):
    """Get saved annotations for a slice."""
    filepath = os.path.join(ANNOTATIONS_DIR, f'{name}.json')

    if not os.path.exists(filepath):
        return jsonify({
            'slice_name': name,
            'boxes': [],
            'timestamp': None
        })

    with open(filepath, 'r') as f:
        return jsonify(json.load(f))


@app.route('/api/annotations/<name>', methods=['POST'])
def save_annotations(name):
    """Save annotations for a slice."""
    data = request.json

    annotations = {
        'slice_name': name,
        'boxes': data.get('boxes', []),
        'timestamp': datetime.now().isoformat()
    }

    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    filepath = os.path.join(ANNOTATIONS_DIR, f'{name}.json')

    with open(filepath, 'w') as f:
        json.dump(annotations, f, indent=2)

    return jsonify({'message': 'Annotations saved', 'path': filepath})


@app.route('/api/generate-ground-truth/<name>', methods=['POST'])
def generate_ground_truth(name):
    """Generate ground truth labels from bounding box annotations."""
    from utils.annotation_utils import boxes_to_labels

    # Load point cloud
    slice_path = os.path.join(SLICES_DIR, f'{name}.las')
    if not os.path.exists(slice_path):
        return jsonify({'error': 'Slice not found'}), 404

    # Load annotations
    ann_path = os.path.join(ANNOTATIONS_DIR, f'{name}.json')
    if not os.path.exists(ann_path):
        return jsonify({'error': 'No annotations found'}), 404

    with open(ann_path, 'r') as f:
        annotations = json.load(f)

    if not annotations.get('boxes'):
        return jsonify({'error': 'No bounding boxes in annotations'}), 400

    # Load labels
    labels = get_labels()

    # Read point cloud
    data = read_las(slice_path)

    # Convert boxes to point labels
    point_labels = boxes_to_labels(data['xyz'], annotations['boxes'], labels['classes'])

    # Save ground truth
    gt_path = os.path.join(GROUND_TRUTH_DIR, f'{name}_gt.las')
    write_las(gt_path, data['xyz'], data['rgb'], point_labels)

    # Count labels
    unique, counts = np.unique(point_labels, return_counts=True)
    label_counts = {}
    for u, c in zip(unique, counts):
        class_name = next((cls['name'] for cls in labels['classes'] if cls['id'] == u), f'class_{u}')
        label_counts[class_name] = int(c)

    return jsonify({
        'message': 'Ground truth generated',
        'path': gt_path,
        'total_points': len(point_labels),
        'label_counts': label_counts
    })


@app.route('/api/predictions/<name>')
def get_predictions(name):
    """Get prediction results for a slice."""
    pred_path = os.path.join(GROUND_TRUTH_DIR, f'{name}_pred.las')

    if not os.path.exists(pred_path):
        return jsonify({'error': 'No predictions found'}), 404

    data = read_las(pred_path)

    # Subsample for display
    xyz, rgb, classification = subsample_points(
        data['xyz'], data['rgb'], data['classification'],
        max_points=MAX_POINTS_DISPLAY
    )

    response = {
        'name': name,
        'num_points': len(xyz),
        'x': xyz[:, 0].tolist(),
        'y': xyz[:, 1].tolist(),
        'z': xyz[:, 2].tolist(),
        'classification': classification.tolist() if classification is not None else []
    }

    return jsonify(response)


@app.route('/api/train', methods=['POST'])
def train_model():
    """Trigger model training."""
    import threading
    from training.train import train

    # Get parameters from request
    data = request.json or {}
    epochs = data.get('epochs', 50)
    batch_size = data.get('batch_size', 16)
    num_points = data.get('num_points', 8192)

    # Check for ground truth files
    gt_files = glob.glob(os.path.join(GROUND_TRUTH_DIR, '*_gt.las'))
    if not gt_files:
        return jsonify({'error': 'No ground truth files found. Generate ground truth first.'}), 400

    # Run training in background thread
    def run_training():
        try:
            result_path = train(
                gt_files=gt_files,
                epochs=epochs,
                batch_size=batch_size,
                num_points=num_points
            )
            print(f"Training complete. Model saved to: {result_path}")
        except Exception as e:
            print(f"Training error: {e}")

    thread = threading.Thread(target=run_training)
    thread.start()

    return jsonify({
        'message': 'Training started',
        'gt_files': len(gt_files),
        'epochs': epochs
    })


@app.route('/api/inference/<name>', methods=['POST'])
def run_inference_endpoint(name):
    """Run inference on a slice."""
    from training.inference import run_inference

    # Check if model exists
    model_files = glob.glob(os.path.join(MODELS_DIR, '*.pth'))
    if not model_files:
        return jsonify({'error': 'No trained model found. Train a model first.'}), 400

    # Check if slice exists
    slice_path = os.path.join(SLICES_DIR, f'{name}.las')
    if not os.path.exists(slice_path):
        return jsonify({'error': f'Slice not found: {name}'}), 404

    try:
        result = run_inference(name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-status')
def model_status():
    """Get current model status."""
    model_files = glob.glob(os.path.join(MODELS_DIR, '*.pth'))
    latest_model = max(model_files, key=os.path.getctime) if model_files else None

    return jsonify({
        'has_model': latest_model is not None,
        'model_path': latest_model,
        'model_count': len(model_files)
    })


# Need numpy for generate_ground_truth
import numpy as np

if __name__ == '__main__':
    print(f"Starting annotation server on http://{WEBAPP_HOST}:{WEBAPP_PORT}")
    print(f"Slices directory: {SLICES_DIR}")
    print(f"Annotations directory: {ANNOTATIONS_DIR}")
    app.run(host=WEBAPP_HOST, port=WEBAPP_PORT, debug=True)
