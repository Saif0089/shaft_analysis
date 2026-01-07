# Building Standalone Windows Executable

This guide explains how to create a standalone `.exe` file that can be distributed to clients without requiring Python installation.

## Prerequisites

- Windows 10/11 (64-bit)
- Python 3.8+ installed
- ~5GB free disk space for build

## Quick Build (Automated)

1. Double-click `build_exe.bat`
2. Wait for the build to complete (~10-15 minutes)
3. Find the executable at `dist/ShaftSegmentation.exe`

## Manual Build Steps

### 1. Set up Build Environment

```cmd
python -m venv build_env
build_env\Scripts\activate
```

### 2. Install Dependencies

```cmd
pip install --upgrade pip
pip install pyinstaller

# Install PyTorch with CUDA support (for GPU acceleration)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Or CPU-only version (smaller file size)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install PyQt5>=5.15.0
pip install numpy>=1.21.0
pip install laspy>=2.0.0
pip install plyfile>=0.7.0
pip install matplotlib>=3.5.0
```

### 3. Build Executable

```cmd
pyinstaller --clean shaft_segmentation.spec
```

### 4. Locate Output

The executable will be at: `dist/ShaftSegmentation.exe`

## Distribution

### Single File Distribution
The `ShaftSegmentation.exe` includes everything needed:
- Python runtime
- All dependencies (PyTorch, PyQt5, etc.)
- Trained model checkpoint
- Model architecture

Simply share the single `.exe` file with clients.

### File Size
- With CUDA support: ~800MB - 1.2GB
- CPU-only build: ~400-600MB

## Troubleshooting

### Build Fails with "torch" errors
Ensure PyTorch is installed correctly:
```cmd
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Antivirus False Positive
PyInstaller executables may trigger false positives. Add an exception or sign the executable.

### Missing DLLs at Runtime
If the exe fails to run on target machine, ensure Visual C++ Redistributable is installed:
https://aka.ms/vs/17/release/vc_redist.x64.exe

### Slow Startup
First launch extracts files to temp directory. Subsequent launches are faster.

## Creating CPU-Only Build (Smaller Size)

To create a smaller executable without GPU support:

1. Install CPU-only PyTorch:
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

2. Build as normal:
```cmd
pyinstaller --clean shaft_segmentation.spec
```

## Signing the Executable (Optional)

For professional distribution, sign the executable to avoid Windows SmartScreen warnings:

```cmd
signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com dist\ShaftSegmentation.exe
```
