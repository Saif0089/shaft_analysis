"""Configuration settings for shaft segmentation project."""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SLICES_DIR = os.path.join(DATA_DIR, 'slices')
ANNOTATIONS_DIR = os.path.join(BASE_DIR, 'annotations')
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, 'ground_truth')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Source slices location
SOURCE_SLICES_DIR = '/home/administrator/shaft_segmentation/shaft_slices_10m'

# Web app settings
WEBAPP_HOST = '0.0.0.0'
WEBAPP_PORT = 8080

# Model settings
NUM_POINTS = 8192  # Points per training sample
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 100

# Point cloud settings
MAX_POINTS_DISPLAY = 500000  # Max points to send to browser at once
POINT_SUBSAMPLE_RATIO = 1.0  # Subsample ratio for display (1.0 = all points)

# Create directories if they don't exist
for dir_path in [DATA_DIR, SLICES_DIR, ANNOTATIONS_DIR, GROUND_TRUTH_DIR, MODELS_DIR]:
    os.makedirs(dir_path, exist_ok=True)
