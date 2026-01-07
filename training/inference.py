"""Inference script for PointNet++ semantic segmentation."""
import os
import sys
import json
import glob
import argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODELS_DIR, SLICES_DIR, GROUND_TRUTH_DIR, ANNOTATIONS_DIR
from training.model import PointNet2SemSeg
from training.dataset import InferenceDataset
from utils.las_io import read_las, write_las


def get_latest_model() -> str:
    """Get path to the latest/best trained model."""
    model_files = glob.glob(os.path.join(MODELS_DIR, 'best_model_*.pth'))
    if not model_files:
        model_files = glob.glob(os.path.join(MODELS_DIR, '*.pth'))
    if not model_files:
        raise FileNotFoundError("No trained model found!")
    return max(model_files, key=os.path.getctime)


def get_num_classes() -> int:
    """Get number of classes from labels file."""
    labels_file = os.path.join(ANNOTATIONS_DIR, 'labels.json')
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            data = json.load(f)
            return max(c['id'] for c in data['classes']) + 1
    return 2


def run_inference(slice_name: str, model_path: str = None,
                  batch_size: int = 16, num_points: int = 8192,
                  device_id: int = 0) -> dict:
    """
    Run inference on a slice.

    Args:
        slice_name: Name of slice (without .las extension)
        model_path: Path to model checkpoint (uses latest if None)
        batch_size: Batch size for inference
        num_points: Points per batch
        device_id: CUDA device ID

    Returns:
        Dictionary with accuracy metrics and output path
    """
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get model path
    if model_path is None:
        model_path = get_latest_model()
    print(f"Using model: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint.get('num_classes', get_num_classes())
    print(f"Number of classes: {num_classes}")

    # Create model
    model = PointNet2SemSeg(num_classes=num_classes, in_channels=3).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load slice
    slice_path = os.path.join(SLICES_DIR, f'{slice_name}.las')
    if not os.path.exists(slice_path):
        raise FileNotFoundError(f"Slice not found: {slice_path}")

    print(f"Loading slice: {slice_path}")
    data = read_las(slice_path)
    all_points = data['xyz']
    print(f"Total points: {len(all_points):,}")

    # Create dataset
    dataset = InferenceDataset(slice_path, num_points=num_points)

    # Accumulate predictions with voting
    predictions_count = np.zeros((len(all_points), num_classes), dtype=np.float32)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Inference"):
            points, indices = dataset[i]
            points = points.unsqueeze(0).to(device)

            outputs = model(points)  # (1, N, num_classes)
            probs = torch.softmax(outputs, dim=-1)
            probs = probs.squeeze(0).cpu().numpy()

            # Accumulate predictions
            for j, idx in enumerate(indices):
                predictions_count[idx] += probs[j]

    # Get final predictions (majority vote)
    predictions = np.argmax(predictions_count, axis=1)

    # Save predictions
    output_path = os.path.join(GROUND_TRUTH_DIR, f'{slice_name}_pred.las')
    write_las(output_path, all_points, data['rgb'], predictions)
    print(f"Saved predictions to: {output_path}")

    # Calculate statistics
    unique, counts = np.unique(predictions, return_counts=True)
    label_counts = {int(u): int(c) for u, c in zip(unique, counts)}

    # If ground truth exists, calculate accuracy
    accuracy = None
    per_class_accuracy = None
    gt_path = os.path.join(GROUND_TRUTH_DIR, f'{slice_name}_gt.las')

    if os.path.exists(gt_path):
        gt_data = read_las(gt_path)
        gt_labels = gt_data['classification']

        if gt_labels is not None and len(gt_labels) == len(predictions):
            accuracy = (predictions == gt_labels).mean() * 100

            # Per-class accuracy
            per_class_accuracy = {}
            for label in np.unique(gt_labels):
                mask = gt_labels == label
                if mask.sum() > 0:
                    per_class_accuracy[int(label)] = (predictions[mask] == gt_labels[mask]).mean() * 100

            print(f"\nAccuracy vs ground truth: {accuracy:.2f}%")
            print(f"Per-class accuracy: {per_class_accuracy}")

    result = {
        'slice_name': slice_name,
        'output_path': output_path,
        'total_points': len(all_points),
        'label_counts': label_counts,
        'accuracy': accuracy,
        'per_class_accuracy': per_class_accuracy
    }

    return result


def batch_inference(slice_names: list = None, model_path: str = None,
                    batch_size: int = 16, num_points: int = 8192,
                    device_id: int = 0) -> list:
    """
    Run inference on multiple slices.

    Args:
        slice_names: List of slice names (if None, process all in SLICES_DIR)
        model_path: Path to model checkpoint
        batch_size: Batch size
        num_points: Points per batch
        device_id: CUDA device ID

    Returns:
        List of result dictionaries
    """
    if slice_names is None:
        slice_files = [f.replace('.las', '') for f in os.listdir(SLICES_DIR)
                       if f.endswith('.las')]
        slice_names = sorted(slice_files)

    results = []
    for slice_name in slice_names:
        print(f"\n{'='*50}")
        print(f"Processing {slice_name}")
        print('='*50)

        try:
            result = run_inference(
                slice_name, model_path, batch_size, num_points, device_id
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing {slice_name}: {e}")
            results.append({
                'slice_name': slice_name,
                'error': str(e)
            })

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on shaft slices')
    parser.add_argument('--slice', type=str, required=True, help='Slice name (without .las)')
    parser.add_argument('--model', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_points', type=int, default=8192, help='Points per batch')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID')

    args = parser.parse_args()

    result = run_inference(
        args.slice,
        model_path=args.model,
        batch_size=args.batch_size,
        num_points=args.num_points,
        device_id=args.device
    )

    print("\nResult:")
    print(json.dumps(result, indent=2))
