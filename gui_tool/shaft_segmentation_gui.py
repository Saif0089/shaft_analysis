"""
Shaft Point Cloud Segmentation GUI Tool
Windows-compatible GUI for processing shaft point clouds with PointNet++ segmentation.

Features:
- Accepts LAS, E57, PLY, PCD point cloud files
- Automatic 10m slice extraction
- Per-slice inference with progress tracking
- Stitched output with classified PLY
- Separate PLY files per class family
- Visualization generation per slice
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import threading
import queue


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).parent / relative_path


def get_default_checkpoint():
    """Find the default checkpoint path"""
    # Check bundled checkpoint first (PyInstaller)
    bundled = get_resource_path('checkpoints/latest.pth')
    if bundled.exists():
        return str(bundled)

    # Check local checkpoints folder
    local = Path(__file__).parent / 'checkpoints' / 'latest.pth'
    if local.exists():
        return str(local)

    # Check training folder
    training = Path(__file__).parent.parent / 'training' / 'checkpoints_best' / 'latest.pth'
    if training.exists():
        return str(training)

    return ""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# GUI imports - try PyQt5 first, fall back to tkinter
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit,
        QGroupBox, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
        QTabWidget, QListWidget, QListWidgetItem, QMessageBox,
        QSplitter, QFrame, QComboBox
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt5.QtGui import QFont, QColor, QPalette, QIcon
    USE_PYQT = True
except ImportError:
    USE_PYQT = False
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Point cloud libraries
try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

try:
    import pye57
    HAS_E57 = True
except ImportError:
    HAS_E57 = False

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import plyfile
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False


# Class name mapping (from training)
CLASS_NAMES = ['wall', 'pipe', 'guard', 'bunton', 'wire', 'column', 'sheet', 'column2']
CLASS_COLORS = [
    (51, 102, 204),    # Blue - wall
    (255, 153, 0),     # Orange - pipe
    (51, 179, 51),     # Green - guard
    (230, 51, 51),     # Red - bunton (guard2)
    (128, 0, 128),     # Purple - wire
    (0, 204, 204),     # Cyan - column
    (230, 230, 51),    # Yellow - sheet
    (128, 128, 128),   # Gray - column2
]

# Map display names for output files
CLASS_FILE_NAMES = {
    'wall': 'walls',
    'pipe': 'pipes',
    'guard': 'guards',
    'bunton': 'buntons',
    'wire': 'wires',
    'column': 'columns',
    'sheet': 'sheets',
    'column2': 'columns_secondary'
}


def load_point_cloud(file_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load point cloud from various formats.
    Returns: points (N,3), colors (N,3) normalized 0-1, intensities (N,) or None
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext == '.las' or ext == '.laz':
        if not HAS_LASPY:
            raise ImportError("laspy required for LAS files. Install with: pip install laspy")
        las = laspy.read(str(file_path))
        points = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)

        # Get colors
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            colors = np.vstack([las.red, las.green, las.blue]).T.astype(np.float32)
            # Normalize based on data range
            if colors.max() > 255:
                colors = colors / 65535.0
            else:
                colors = colors / 255.0
        else:
            colors = None

        # Get intensity
        intensity = np.array(las.intensity).astype(np.float32) if hasattr(las, 'intensity') else None

        return points, colors, intensity

    elif ext == '.e57':
        if not HAS_E57:
            raise ImportError("pye57 required for E57 files. Install with: pip install pye57")
        e57 = pye57.E57(str(file_path))
        data = e57.read_scan(0)
        points = np.column_stack([data['cartesianX'], data['cartesianY'], data['cartesianZ']])

        # Colors
        if 'colorRed' in data:
            colors = np.column_stack([data['colorRed'], data['colorGreen'], data['colorBlue']]) / 255.0
        else:
            colors = None

        intensity = data.get('intensity', None)
        return points, colors, intensity

    elif ext == '.ply':
        if HAS_OPEN3D:
            pcd = o3d.io.read_point_cloud(str(file_path))
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None
            return points, colors, None
        elif HAS_PLYFILE:
            plydata = plyfile.PlyData.read(str(file_path))
            vertex = plydata['vertex']
            points = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
            if 'red' in vertex:
                colors = np.column_stack([vertex['red'], vertex['green'], vertex['blue']]) / 255.0
            else:
                colors = None
            return points, colors, None
        else:
            raise ImportError("open3d or plyfile required for PLY files")

    elif ext == '.pcd':
        if not HAS_OPEN3D:
            raise ImportError("open3d required for PCD files. Install with: pip install open3d")
        pcd = o3d.io.read_point_cloud(str(file_path))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        return points, colors, None

    else:
        raise ValueError(f"Unsupported file format: {ext}")


def save_ply(file_path: str, points: np.ndarray, colors: np.ndarray = None,
             classifications: np.ndarray = None):
    """Save point cloud to PLY format with colors."""
    n_points = len(points)

    with open(file_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        if classifications is not None:
            f.write("property uchar classification\n")
        f.write("end_header\n")

        for i in range(n_points):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
            if colors is not None:
                r, g, b = colors[i]
                if isinstance(r, float) and r <= 1.0:
                    r, g, b = int(r * 255), int(g * 255), int(b * 255)
                line += f" {int(r)} {int(g)} {int(b)}"
            if classifications is not None:
                line += f" {int(classifications[i])}"
            f.write(line + "\n")


class SliceInferenceDataset(Dataset):
    """Dataset for inference on a single slice."""

    def __init__(self, points: np.ndarray, colors: np.ndarray,
                 num_points: int = 4096, block_size: float = 1.0, stride: float = 0.5):
        self.points = points.astype(np.float32)
        self.colors = colors.astype(np.float32) if colors is not None else np.ones((len(points), 3), dtype=np.float32) * 0.148
        self.num_points = num_points
        self.block_size = block_size
        self.stride = stride

        self.blocks = self._create_blocks()

    def _create_blocks(self) -> List[Tuple[float, float, np.ndarray]]:
        blocks = []
        min_bound = self.points.min(axis=0)
        max_bound = self.points.max(axis=0)

        x_range = np.arange(min_bound[0], max_bound[0], self.stride)
        y_range = np.arange(min_bound[1], max_bound[1], self.stride)

        for x in x_range:
            for y in y_range:
                mask = (
                    (self.points[:, 0] >= x) & (self.points[:, 0] < x + self.block_size) &
                    (self.points[:, 1] >= y) & (self.points[:, 1] < y + self.block_size)
                )
                indices = np.where(mask)[0]

                if len(indices) >= 10:
                    blocks.append((x, y, indices))

        return blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x, y, point_indices = self.blocks[idx]

        block_points = self.points[point_indices].copy()
        block_colors = self.colors[point_indices].copy()
        original_indices = point_indices.copy()

        if len(block_points) > self.num_points:
            choice = np.random.choice(len(block_points), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(block_points), self.num_points, replace=True)

        sampled_points = block_points[choice]
        sampled_colors = block_colors[choice]
        sampled_indices = original_indices[choice]

        block_center = np.array([
            x + self.block_size / 2,
            y + self.block_size / 2,
            sampled_points[:, 2].mean()
        ])
        sampled_points = sampled_points - block_center

        return {
            'points': torch.from_numpy(sampled_points).float(),
            'features': torch.from_numpy(sampled_colors).float(),
            'indices': torch.from_numpy(sampled_indices).long(),
            'block_center': torch.from_numpy(block_center).float()
        }


class ShaftSegmentationProcessor:
    """Main processing engine for shaft segmentation."""

    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.config = None
        self.class_names = CLASS_NAMES
        self.num_classes = len(CLASS_NAMES)

    def load_model(self, progress_callback=None):
        """Load the model from checkpoint."""
        if progress_callback:
            progress_callback("Loading model checkpoint...")

        # Try bundled models first (PyInstaller), then local, then training directory
        try:
            # Try bundled path first
            models_path = get_resource_path('models')
            if models_path.exists():
                sys.path.insert(0, str(models_path.parent))
            from models.pointnet2 import PointNet2Segmentation, PointNet2SegmentationMSG, PointNet2SegmentationLight
        except ImportError:
            # Add training directory to path for imports
            training_dir = Path(self.checkpoint_path).parent.parent
            sys.path.insert(0, str(training_dir))
            from models.pointnet2 import PointNet2Segmentation, PointNet2SegmentationMSG, PointNet2SegmentationLight

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        self.config = checkpoint['config']
        dataset_info = checkpoint['dataset_info']
        self.num_classes = dataset_info['num_classes']
        self.class_names = dataset_info['class_names']

        # Fix class name mapping (guard2 -> bunton)
        self.class_names = [name if name != 'guard2' else 'bunton' for name in self.class_names]

        # Create model
        if self.config.get('use_light', False):
            self.model = PointNet2SegmentationLight(num_classes=self.num_classes, in_channels=3)
        elif self.config.get('use_msg', False):
            self.model = PointNet2SegmentationMSG(num_classes=self.num_classes, in_channels=3)
        else:
            self.model = PointNet2Segmentation(num_classes=self.num_classes, in_channels=3)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        if progress_callback:
            progress_callback(f"Model loaded. Classes: {self.class_names}")

    def extract_slices(self, points: np.ndarray, colors: np.ndarray,
                       intensity: np.ndarray, slice_height: float = 10.0,
                       progress_callback=None) -> List[Dict]:
        """Extract 10m vertical slices from the point cloud."""
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        total_height = z_max - z_min

        slices = []
        current_z = z_min
        slice_idx = 0

        while current_z < z_max:
            slice_z_max = min(current_z + slice_height, z_max)

            # Get points in this slice
            mask = (points[:, 2] >= current_z) & (points[:, 2] < slice_z_max)
            slice_points = points[mask]
            slice_colors = colors[mask] if colors is not None else None
            slice_intensity = intensity[mask] if intensity is not None else None

            if len(slice_points) > 100:  # Skip empty slices
                slices.append({
                    'index': slice_idx,
                    'z_min': current_z,
                    'z_max': slice_z_max,
                    'points': slice_points,
                    'colors': slice_colors,
                    'intensity': slice_intensity,
                    'original_indices': np.where(mask)[0]
                })

                if progress_callback:
                    progress_callback(f"Extracted slice {slice_idx}: z=[{current_z:.1f}, {slice_z_max:.1f}], {len(slice_points):,} points")

            slice_idx += 1
            current_z = slice_z_max

        return slices

    @torch.no_grad()
    def process_slice(self, slice_data: Dict, batch_size: int = 32,
                      progress_callback=None) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on a single slice."""
        points = slice_data['points']
        colors = slice_data['colors']
        intensity = slice_data['intensity']

        # Prepare colors
        if colors is not None:
            features = colors.copy()
        elif intensity is not None:
            # Normalize intensity to match training distribution
            norm_intensity = intensity / (intensity.mean() + 1e-8) * 0.149
            norm_intensity = np.clip(norm_intensity, 0, 1)
            features = np.column_stack([norm_intensity, norm_intensity, norm_intensity])
        else:
            features = np.ones((len(points), 3), dtype=np.float32) * 0.148

        # Create dataset
        dataset = SliceInferenceDataset(points, features, num_points=4096, block_size=1.0, stride=0.5)

        if len(dataset) == 0:
            if progress_callback:
                progress_callback(f"  Warning: No valid blocks in slice")
            return np.zeros(len(points), dtype=np.int32), np.zeros(len(points), dtype=np.float32)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        # Initialize vote counts
        num_points = len(points)
        vote_counts = np.zeros((num_points, self.num_classes), dtype=np.float32)

        for batch_idx, batch in enumerate(loader):
            pts = batch['points'].to(self.device)
            feats = batch['features'].to(self.device)
            indices = batch['indices'].numpy()

            outputs = self.model(pts, feats)
            probs = F.softmax(outputs, dim=-1).cpu().numpy()

            B, N, C = probs.shape
            for b in range(B):
                batch_indices = indices[b]
                batch_probs = probs[b]

                for n in range(N):
                    idx = batch_indices[n]
                    if idx < num_points:
                        vote_counts[idx] += batch_probs[n]

        # Finalize predictions
        predictions = vote_counts.argmax(axis=-1).astype(np.int32)
        confidences = vote_counts.max(axis=-1) / (vote_counts.sum(axis=-1) + 1e-8)

        return predictions, confidences

    def create_slice_visualization(self, slice_data: Dict, predictions: np.ndarray,
                                   output_path: str, sample_size: int = 50000):
        """Create visualization for a slice."""
        points = slice_data['points']

        if len(points) > sample_size:
            indices = np.random.choice(len(points), sample_size, replace=False)
            plot_points = points[indices]
            plot_preds = predictions[indices]
        else:
            plot_points = points
            plot_preds = predictions

        colors = np.array(CLASS_COLORS) / 255.0

        fig = plt.figure(figsize=(20, 15))

        # Top view
        ax1 = fig.add_subplot(2, 2, 1)
        for cls in range(self.num_classes):
            mask = plot_preds == cls
            if mask.any():
                name = self.class_names[cls]
                ax1.scatter(plot_points[mask, 0], plot_points[mask, 1],
                           s=0.5, alpha=0.5, c=[colors[cls % len(colors)]],
                           label=f'{name} ({mask.sum():,})')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title(f'Top View (XY) - Slice {slice_data["index"]}')
        ax1.legend(markerscale=10, fontsize=8)
        ax1.set_aspect('equal')

        # Side view XZ
        ax2 = fig.add_subplot(2, 2, 2)
        for cls in range(self.num_classes):
            mask = plot_preds == cls
            if mask.any():
                ax2.scatter(plot_points[mask, 0], plot_points[mask, 2],
                           s=0.5, alpha=0.5, c=[colors[cls % len(colors)]])
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title('Side View (XZ)')

        # Side view YZ
        ax3 = fig.add_subplot(2, 2, 3)
        for cls in range(self.num_classes):
            mask = plot_preds == cls
            if mask.any():
                ax3.scatter(plot_points[mask, 1], plot_points[mask, 2],
                           s=0.5, alpha=0.5, c=[colors[cls % len(colors)]])
        ax3.set_xlabel('Y (m)')
        ax3.set_ylabel('Z (m)')
        ax3.set_title('Side View (YZ)')

        # Class distribution
        ax4 = fig.add_subplot(2, 2, 4)
        class_counts = [(predictions == i).sum() for i in range(self.num_classes)]
        bars = ax4.bar(self.class_names, class_counts,
                       color=[colors[i % len(colors)] for i in range(self.num_classes)])
        ax4.set_xlabel('Class')
        ax4.set_ylabel('Point Count')
        ax4.set_title('Class Distribution')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


class ProcessingThread(QThread if USE_PYQT else threading.Thread):
    """Background thread for processing."""

    if USE_PYQT:
        progress_signal = pyqtSignal(str)
        progress_value_signal = pyqtSignal(int)
        finished_signal = pyqtSignal(bool, str)

    def __init__(self, processor, input_path, output_dir, slice_height=10.0):
        if USE_PYQT:
            super().__init__()
        else:
            super().__init__()
            self.progress_queue = queue.Queue()

        self.processor = processor
        self.input_path = input_path
        self.output_dir = output_dir
        self.slice_height = slice_height
        self.cancelled = False

    def emit_progress(self, msg):
        if USE_PYQT:
            self.progress_signal.emit(msg)
        else:
            self.progress_queue.put(('msg', msg))

    def emit_progress_value(self, value):
        if USE_PYQT:
            self.progress_value_signal.emit(value)
        else:
            self.progress_queue.put(('value', value))

    def emit_finished(self, success, msg):
        if USE_PYQT:
            self.finished_signal.emit(success, msg)
        else:
            self.progress_queue.put(('finished', (success, msg)))

    def run(self):
        try:
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories
            vis_dir = output_dir / "visualizations"
            families_dir = output_dir / "classified_families"
            vis_dir.mkdir(exist_ok=True)
            families_dir.mkdir(exist_ok=True)

            # Load point cloud
            self.emit_progress(f"Loading point cloud: {self.input_path}")
            points, colors, intensity = load_point_cloud(self.input_path)
            self.emit_progress(f"Loaded {len(points):,} points")

            # Prepare colors for processing
            if colors is None and intensity is not None:
                norm_intensity = intensity / (intensity.mean() + 1e-8) * 0.149
                norm_intensity = np.clip(norm_intensity, 0, 1)
                colors = np.column_stack([norm_intensity, norm_intensity, norm_intensity])
            elif colors is None:
                colors = np.ones((len(points), 3), dtype=np.float32) * 0.148

            # Extract slices
            self.emit_progress("Extracting vertical slices...")
            slices = self.processor.extract_slices(points, colors, intensity,
                                                   self.slice_height, self.emit_progress)
            self.emit_progress(f"Extracted {len(slices)} slices")

            # Initialize combined predictions array
            all_predictions = np.zeros(len(points), dtype=np.int32)
            all_confidences = np.zeros(len(points), dtype=np.float32)

            # Process each slice
            for i, slice_data in enumerate(slices):
                if self.cancelled:
                    self.emit_finished(False, "Processing cancelled")
                    return

                self.emit_progress(f"\nProcessing slice {i+1}/{len(slices)} (z={slice_data['z_min']:.1f}m to {slice_data['z_max']:.1f}m)")
                self.emit_progress_value(int((i / len(slices)) * 100))

                # Run inference
                predictions, confidences = self.processor.process_slice(slice_data, progress_callback=self.emit_progress)

                # Store predictions in original indices
                original_indices = slice_data['original_indices']
                all_predictions[original_indices] = predictions
                all_confidences[original_indices] = confidences

                # Print class distribution
                for cls in range(self.processor.num_classes):
                    count = (predictions == cls).sum()
                    name = self.processor.class_names[cls]
                    self.emit_progress(f"  {name}: {count:,} points ({100*count/len(predictions):.1f}%)")

                # Create visualization
                vis_path = vis_dir / f"slice_{i:03d}_z{slice_data['z_min']:.0f}_{slice_data['z_max']:.0f}.png"
                self.processor.create_slice_visualization(slice_data, predictions, str(vis_path))
                self.emit_progress(f"  Saved visualization: {vis_path.name}")

            self.emit_progress_value(80)

            # Generate classified colors
            self.emit_progress("\nGenerating classified point cloud...")
            classified_colors = np.zeros((len(points), 3), dtype=np.uint8)
            for cls in range(self.processor.num_classes):
                mask = all_predictions == cls
                classified_colors[mask] = CLASS_COLORS[cls]

            # Save combined PLY with classified colors
            combined_ply_path = output_dir / "combined_classified.ply"
            save_ply(str(combined_ply_path), points, classified_colors / 255.0, all_predictions)
            self.emit_progress(f"Saved combined classified PLY: {combined_ply_path}")

            # Save per-class PLY files
            self.emit_progress("\nSaving per-class PLY files...")
            for cls in range(self.processor.num_classes):
                mask = all_predictions == cls
                if mask.sum() > 0:
                    class_name = self.processor.class_names[cls]
                    file_name = CLASS_FILE_NAMES.get(class_name, class_name)
                    class_points = points[mask]
                    class_colors = classified_colors[mask]

                    class_ply_path = families_dir / f"{file_name}.ply"
                    save_ply(str(class_ply_path), class_points, class_colors / 255.0)
                    self.emit_progress(f"  Saved {file_name}.ply ({mask.sum():,} points)")

            # Save metadata
            metadata = {
                'input_file': str(self.input_path),
                'total_points': len(points),
                'num_slices': len(slices),
                'slice_height_m': self.slice_height,
                'class_distribution': {
                    self.processor.class_names[i]: int((all_predictions == i).sum())
                    for i in range(self.processor.num_classes)
                },
                'mean_confidence': float(all_confidences.mean()),
                'processing_date': datetime.now().isoformat()
            }

            with open(output_dir / "processing_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            self.emit_progress_value(100)
            self.emit_finished(True, f"Processing complete! Output saved to: {output_dir}")

        except Exception as e:
            import traceback
            self.emit_finished(False, f"Error: {str(e)}\n{traceback.format_exc()}")


if USE_PYQT:
    class ShaftSegmentationGUI(QMainWindow):
        """PyQt5 GUI for shaft segmentation."""

        def __init__(self):
            super().__init__()
            self.processor = None
            self.processing_thread = None
            self.init_ui()

        def init_ui(self):
            self.setWindowTitle("Shaft Point Cloud Segmentation Tool")
            self.setMinimumSize(900, 700)

            # Central widget
            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)

            # Input section
            input_group = QGroupBox("Input")
            input_layout = QVBoxLayout(input_group)

            # Point cloud file
            pc_layout = QHBoxLayout()
            pc_layout.addWidget(QLabel("Point Cloud:"))
            self.pc_path_edit = QLineEdit()
            self.pc_path_edit.setPlaceholderText("Select LAS, E57, PLY, or PCD file...")
            pc_layout.addWidget(self.pc_path_edit)
            self.pc_browse_btn = QPushButton("Browse...")
            self.pc_browse_btn.clicked.connect(self.browse_point_cloud)
            pc_layout.addWidget(self.pc_browse_btn)
            input_layout.addLayout(pc_layout)

            # Checkpoint file
            ckpt_layout = QHBoxLayout()
            ckpt_layout.addWidget(QLabel("Model Checkpoint:"))
            self.ckpt_path_edit = QLineEdit()
            # Default checkpoint path
            default_ckpt = get_default_checkpoint()
            if default_ckpt:
                self.ckpt_path_edit.setText(default_ckpt)
            ckpt_layout.addWidget(self.ckpt_path_edit)
            self.ckpt_browse_btn = QPushButton("Browse...")
            self.ckpt_browse_btn.clicked.connect(self.browse_checkpoint)
            ckpt_layout.addWidget(self.ckpt_browse_btn)
            input_layout.addLayout(ckpt_layout)

            layout.addWidget(input_group)

            # Output section
            output_group = QGroupBox("Output")
            output_layout = QVBoxLayout(output_group)

            out_layout = QHBoxLayout()
            out_layout.addWidget(QLabel("Output Directory:"))
            self.out_path_edit = QLineEdit()
            self.out_path_edit.setPlaceholderText("Select output directory...")
            out_layout.addWidget(self.out_path_edit)
            self.out_browse_btn = QPushButton("Browse...")
            self.out_browse_btn.clicked.connect(self.browse_output)
            out_layout.addWidget(self.out_browse_btn)
            output_layout.addLayout(out_layout)

            layout.addWidget(output_group)

            # Settings section
            settings_group = QGroupBox("Settings")
            settings_layout = QHBoxLayout(settings_group)

            settings_layout.addWidget(QLabel("Slice Height (m):"))
            self.slice_height_spin = QDoubleSpinBox()
            self.slice_height_spin.setRange(1.0, 100.0)
            self.slice_height_spin.setValue(10.0)
            self.slice_height_spin.setSingleStep(1.0)
            settings_layout.addWidget(self.slice_height_spin)

            settings_layout.addStretch()

            settings_layout.addWidget(QLabel("Device:"))
            self.device_combo = QComboBox()
            self.device_combo.addItem("Auto (GPU if available)")
            self.device_combo.addItem("CPU")
            if torch.cuda.is_available():
                self.device_combo.addItem("CUDA")
            settings_layout.addWidget(self.device_combo)

            layout.addWidget(settings_group)

            # Progress section
            progress_group = QGroupBox("Progress")
            progress_layout = QVBoxLayout(progress_group)

            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            progress_layout.addWidget(self.progress_bar)

            self.log_text = QTextEdit()
            self.log_text.setReadOnly(True)
            self.log_text.setFont(QFont("Consolas", 9))
            self.log_text.setMinimumHeight(250)
            progress_layout.addWidget(self.log_text)

            layout.addWidget(progress_group)

            # Buttons
            btn_layout = QHBoxLayout()

            self.process_btn = QPushButton("Start Processing")
            self.process_btn.setMinimumHeight(40)
            self.process_btn.clicked.connect(self.start_processing)
            btn_layout.addWidget(self.process_btn)

            self.cancel_btn = QPushButton("Cancel")
            self.cancel_btn.setMinimumHeight(40)
            self.cancel_btn.setEnabled(False)
            self.cancel_btn.clicked.connect(self.cancel_processing)
            btn_layout.addWidget(self.cancel_btn)

            layout.addLayout(btn_layout)

        def browse_point_cloud(self):
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Point Cloud File", "",
                "Point Clouds (*.las *.laz *.e57 *.ply *.pcd);;All Files (*.*)"
            )
            if file_path:
                self.pc_path_edit.setText(file_path)
                # Auto-set output directory
                if not self.out_path_edit.text():
                    out_dir = Path(file_path).parent / f"{Path(file_path).stem}_segmented"
                    self.out_path_edit.setText(str(out_dir))

        def browse_checkpoint(self):
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Model Checkpoint", "",
                "PyTorch Checkpoints (*.pth *.pt);;All Files (*.*)"
            )
            if file_path:
                self.ckpt_path_edit.setText(file_path)

        def browse_output(self):
            dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
            if dir_path:
                self.out_path_edit.setText(dir_path)

        def log(self, message):
            self.log_text.append(message)
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum()
            )

        def start_processing(self):
            # Validate inputs
            pc_path = self.pc_path_edit.text()
            ckpt_path = self.ckpt_path_edit.text()
            out_path = self.out_path_edit.text()

            if not pc_path or not Path(pc_path).exists():
                QMessageBox.warning(self, "Error", "Please select a valid point cloud file.")
                return

            if not ckpt_path or not Path(ckpt_path).exists():
                QMessageBox.warning(self, "Error", "Please select a valid model checkpoint.")
                return

            if not out_path:
                QMessageBox.warning(self, "Error", "Please select an output directory.")
                return

            # Get device
            device_text = self.device_combo.currentText()
            if "CUDA" in device_text:
                device = 'cuda'
            elif "CPU" in device_text:
                device = 'cpu'
            else:
                device = None  # Auto

            # Create processor
            self.processor = ShaftSegmentationProcessor(ckpt_path, device)

            try:
                self.processor.load_model(self.log)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
                return

            # Start processing thread
            self.processing_thread = ProcessingThread(
                self.processor, pc_path, out_path,
                self.slice_height_spin.value()
            )
            self.processing_thread.progress_signal.connect(self.log)
            self.processing_thread.progress_value_signal.connect(self.progress_bar.setValue)
            self.processing_thread.finished_signal.connect(self.processing_finished)

            self.process_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
            self.progress_bar.setValue(0)
            self.log_text.clear()

            self.processing_thread.start()

        def cancel_processing(self):
            if self.processing_thread:
                self.processing_thread.cancelled = True
                self.log("Cancelling...")

        def processing_finished(self, success, message):
            self.process_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)

            if success:
                self.log("\n" + "=" * 50)
                self.log(message)
                QMessageBox.information(self, "Success", message)
            else:
                self.log("\n" + "=" * 50)
                self.log(f"FAILED: {message}")
                QMessageBox.critical(self, "Error", message)

else:
    # Tkinter fallback GUI
    class ShaftSegmentationGUI:
        """Tkinter GUI for shaft segmentation."""

        def __init__(self):
            self.root = tk.Tk()
            self.root.title("Shaft Point Cloud Segmentation Tool")
            self.root.geometry("900x700")

            self.processor = None
            self.processing_thread = None

            self.init_ui()

        def init_ui(self):
            # Main frame
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Input section
            input_frame = ttk.LabelFrame(main_frame, text="Input", padding="5")
            input_frame.pack(fill=tk.X, pady=5)

            # Point cloud
            pc_frame = ttk.Frame(input_frame)
            pc_frame.pack(fill=tk.X, pady=2)
            ttk.Label(pc_frame, text="Point Cloud:").pack(side=tk.LEFT)
            self.pc_path_var = tk.StringVar()
            ttk.Entry(pc_frame, textvariable=self.pc_path_var, width=60).pack(side=tk.LEFT, padx=5)
            ttk.Button(pc_frame, text="Browse...", command=self.browse_point_cloud).pack(side=tk.LEFT)

            # Checkpoint
            ckpt_frame = ttk.Frame(input_frame)
            ckpt_frame.pack(fill=tk.X, pady=2)
            ttk.Label(ckpt_frame, text="Checkpoint:").pack(side=tk.LEFT)
            self.ckpt_path_var = tk.StringVar()
            default_ckpt = get_default_checkpoint()
            if default_ckpt:
                self.ckpt_path_var.set(default_ckpt)
            ttk.Entry(ckpt_frame, textvariable=self.ckpt_path_var, width=60).pack(side=tk.LEFT, padx=5)
            ttk.Button(ckpt_frame, text="Browse...", command=self.browse_checkpoint).pack(side=tk.LEFT)

            # Output section
            output_frame = ttk.LabelFrame(main_frame, text="Output", padding="5")
            output_frame.pack(fill=tk.X, pady=5)

            out_frame = ttk.Frame(output_frame)
            out_frame.pack(fill=tk.X, pady=2)
            ttk.Label(out_frame, text="Output Dir:").pack(side=tk.LEFT)
            self.out_path_var = tk.StringVar()
            ttk.Entry(out_frame, textvariable=self.out_path_var, width=60).pack(side=tk.LEFT, padx=5)
            ttk.Button(out_frame, text="Browse...", command=self.browse_output).pack(side=tk.LEFT)

            # Settings
            settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="5")
            settings_frame.pack(fill=tk.X, pady=5)

            ttk.Label(settings_frame, text="Slice Height (m):").pack(side=tk.LEFT)
            self.slice_height_var = tk.DoubleVar(value=10.0)
            ttk.Spinbox(settings_frame, textvariable=self.slice_height_var,
                       from_=1.0, to=100.0, width=10).pack(side=tk.LEFT, padx=5)

            # Progress
            progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="5")
            progress_frame.pack(fill=tk.BOTH, expand=True, pady=5)

            self.progress_var = tk.IntVar()
            self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                                maximum=100)
            self.progress_bar.pack(fill=tk.X, pady=5)

            self.log_text = scrolledtext.ScrolledText(progress_frame, height=15)
            self.log_text.pack(fill=tk.BOTH, expand=True)

            # Buttons
            btn_frame = ttk.Frame(main_frame)
            btn_frame.pack(fill=tk.X, pady=10)

            self.process_btn = ttk.Button(btn_frame, text="Start Processing",
                                          command=self.start_processing)
            self.process_btn.pack(side=tk.LEFT, padx=5)

            self.cancel_btn = ttk.Button(btn_frame, text="Cancel",
                                         command=self.cancel_processing, state=tk.DISABLED)
            self.cancel_btn.pack(side=tk.LEFT, padx=5)

        def browse_point_cloud(self):
            file_path = filedialog.askopenfilename(
                filetypes=[("Point Clouds", "*.las *.laz *.e57 *.ply *.pcd"), ("All Files", "*.*")]
            )
            if file_path:
                self.pc_path_var.set(file_path)
                if not self.out_path_var.get():
                    out_dir = Path(file_path).parent / f"{Path(file_path).stem}_segmented"
                    self.out_path_var.set(str(out_dir))

        def browse_checkpoint(self):
            file_path = filedialog.askopenfilename(
                filetypes=[("PyTorch Checkpoints", "*.pth *.pt"), ("All Files", "*.*")]
            )
            if file_path:
                self.ckpt_path_var.set(file_path)

        def browse_output(self):
            dir_path = filedialog.askdirectory()
            if dir_path:
                self.out_path_var.set(dir_path)

        def log(self, message):
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)

        def check_thread_queue(self):
            if self.processing_thread and hasattr(self.processing_thread, 'progress_queue'):
                try:
                    while True:
                        msg_type, msg = self.processing_thread.progress_queue.get_nowait()
                        if msg_type == 'msg':
                            self.log(msg)
                        elif msg_type == 'value':
                            self.progress_var.set(msg)
                        elif msg_type == 'finished':
                            success, message = msg
                            self.processing_finished(success, message)
                            return
                except queue.Empty:
                    pass
                self.root.after(100, self.check_thread_queue)

        def start_processing(self):
            pc_path = self.pc_path_var.get()
            ckpt_path = self.ckpt_path_var.get()
            out_path = self.out_path_var.get()

            if not pc_path or not Path(pc_path).exists():
                messagebox.showwarning("Error", "Please select a valid point cloud file.")
                return

            if not ckpt_path or not Path(ckpt_path).exists():
                messagebox.showwarning("Error", "Please select a valid model checkpoint.")
                return

            if not out_path:
                messagebox.showwarning("Error", "Please select an output directory.")
                return

            self.processor = ShaftSegmentationProcessor(ckpt_path)

            try:
                self.processor.load_model(self.log)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                return

            self.processing_thread = ProcessingThread(
                self.processor, pc_path, out_path,
                self.slice_height_var.get()
            )

            self.process_btn.config(state=tk.DISABLED)
            self.cancel_btn.config(state=tk.NORMAL)
            self.progress_var.set(0)
            self.log_text.delete(1.0, tk.END)

            self.processing_thread.start()
            self.root.after(100, self.check_thread_queue)

        def cancel_processing(self):
            if self.processing_thread:
                self.processing_thread.cancelled = True
                self.log("Cancelling...")

        def processing_finished(self, success, message):
            self.process_btn.config(state=tk.NORMAL)
            self.cancel_btn.config(state=tk.DISABLED)

            if success:
                self.log("\n" + "=" * 50)
                self.log(message)
                messagebox.showinfo("Success", message)
            else:
                self.log("\n" + "=" * 50)
                self.log(f"FAILED: {message}")
                messagebox.showerror("Error", message)

        def run(self):
            self.root.mainloop()


def main():
    if USE_PYQT:
        app = QApplication(sys.argv)
        window = ShaftSegmentationGUI()
        window.show()
        sys.exit(app.exec_())
    else:
        gui = ShaftSegmentationGUI()
        gui.run()


if __name__ == '__main__':
    main()
