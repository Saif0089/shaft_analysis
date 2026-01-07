@echo off
REM Shaft Segmentation GUI Launcher for Windows
REM

echo ============================================
echo  Shaft Point Cloud Segmentation Tool
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat

    echo Installing dependencies...
    pip install -r requirements.txt

    REM Install PyTorch with CUDA support if available
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
)

echo.
echo Starting GUI...
python shaft_segmentation_gui.py

pause
