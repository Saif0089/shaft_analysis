# PyInstaller runtime hook for PyTorch
# This adds torch/lib to the DLL search path before torch is imported

import os
import sys

# Get the path to the bundled application
if getattr(sys, 'frozen', False):
    # Running as compiled
    bundle_dir = sys._MEIPASS
else:
    bundle_dir = os.path.dirname(os.path.abspath(__file__))

# Add torch lib directory to DLL search path
torch_lib_path = os.path.join(bundle_dir, 'torch', 'lib')
if os.path.exists(torch_lib_path):
    # For Windows, add to PATH and use os.add_dll_directory (Python 3.8+)
    os.environ['PATH'] = torch_lib_path + os.pathsep + os.environ.get('PATH', '')

    # Python 3.8+ on Windows needs explicit DLL directory
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(torch_lib_path)
