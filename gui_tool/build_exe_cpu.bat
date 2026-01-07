@echo off
REM Build script for CPU-ONLY Windows executable (smaller size, no GPU required)
REM Run this on a Windows machine with Python installed

echo ============================================
echo  Building Shaft Segmentation Tool (CPU-Only)
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Create virtual environment if not exists
if not exist "build_env_cpu\Scripts\activate.bat" (
    echo Creating build environment...
    python -m venv build_env_cpu
)

REM Activate environment
call build_env_cpu\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies (CPU-only PyTorch)...
pip install --upgrade pip
pip install pyinstaller
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install PyQt5>=5.15.0
pip install numpy>=1.21.0
pip install laspy>=2.0.0
pip install plyfile>=0.7.0
pip install matplotlib>=3.5.0

REM Build executable
echo.
echo Building executable...
pyinstaller --clean shaft_segmentation.spec

REM Check if build succeeded
if exist "dist\ShaftSegmentation.exe" (
    echo.
    echo ============================================
    echo  BUILD SUCCESSFUL! (CPU-Only Version)
    echo  Executable: dist\ShaftSegmentation.exe
    echo  Size: Smaller than GPU version
    echo  Works on: Any Windows PC (no GPU needed)
    echo ============================================
) else (
    echo.
    echo BUILD FAILED! Check the error messages above.
)

pause
