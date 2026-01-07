# Shaft Point Cloud Segmentation Tool

Windows-compatible GUI and CLI tool for processing shaft point clouds with PointNet++ deep learning segmentation.

## Features

- **Multi-format support**: LAS, E57, PLY, PCD point cloud files
- **Automatic slicing**: Processes large point clouds in 10m vertical slices
- **8-class segmentation**: wall, pipe, guard, bunton, wire, column, sheet, column2
- **Combined output**: Single PLY with all classifications colored
- **Per-class export**: Separate PLY files for each class family
- **Visualizations**: PNG visualization for each slice

## Installation

### Windows

1. Install Python 3.8+ from https://www.python.org/downloads/
2. Double-click `run_gui.bat` to automatically set up the environment and launch

### Manual Installation

```bash
pip install -r requirements.txt
```

For GPU support (recommended):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### GUI Mode

```bash
python shaft_segmentation_gui.py
```

Or on Windows, double-click `run_gui.bat`

### CLI Mode

```bash
python shaft_segmentation_cli.py --input shaft.las --output output_dir/ --checkpoint checkpoints/latest.pth
```

#### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input`, `-i` | Input point cloud file (required) | - |
| `--output`, `-o` | Output directory (required) | - |
| `--checkpoint`, `-c` | Model checkpoint path | Auto-detect |
| `--slice-height` | Height of each slice in meters | 10.0 |
| `--device` | Device (cuda/cpu) | Auto |
| `--batch-size` | Batch size for inference | 32 |

## Output Structure

```
output_dir/
├── combined_classified.ply      # All points with classification colors
├── processing_metadata.json     # Processing info and statistics
├── classified_families/
│   ├── walls.ply
│   ├── pipes.ply
│   ├── guards.ply
│   ├── buntons.ply
│   ├── wires.ply
│   ├── columns.ply
│   ├── sheets.ply
│   └── columns_secondary.ply
└── visualizations/
    ├── slice_000_z*.png
    ├── slice_001_z*.png
    └── ...
```

## Class Colors

| Class | Color | RGB |
|-------|-------|-----|
| wall | Blue | (51, 102, 204) |
| pipe | Orange | (255, 153, 0) |
| guard | Green | (51, 179, 51) |
| bunton | Red | (230, 51, 51) |
| wire | Purple | (128, 0, 128) |
| column | Cyan | (0, 204, 204) |
| sheet | Yellow | (230, 230, 51) |
| column2 | Gray | (128, 128, 128) |

## Requirements

- Python 3.8+
- PyTorch 1.10+
- PyQt5 (for GUI) or tkinter fallback
- laspy (for LAS files)
- open3d or plyfile (for PLY files)
- matplotlib (for visualizations)

## Model

The tool uses a PointNet++ segmentation model trained on manually annotated shaft point cloud data. The model achieves ~90% accuracy on 8-class segmentation.

## License

Internal use only.
