"""
Launcher script for ShaftSegmentation
This script sets up DLL paths before importing torch
"""
import os
import sys
from pathlib import Path

def setup_dll_paths():
    """Set up DLL search paths for PyTorch before importing it"""
    if getattr(sys, 'frozen', False):
        # Running as compiled exe
        # sys._MEIPASS is the temp folder where PyInstaller extracts files
        bundle_dir = Path(sys._MEIPASS)

        # Add all potential DLL locations to PATH
        dll_paths = [
            bundle_dir,
            bundle_dir / 'torch' / 'lib',
            bundle_dir / 'torch',
        ]

        # Build new PATH with DLL directories first
        new_path_parts = []
        for p in dll_paths:
            if p.exists():
                new_path_parts.append(str(p))

        # Prepend to existing PATH
        existing_path = os.environ.get('PATH', '')
        os.environ['PATH'] = os.pathsep.join(new_path_parts + [existing_path])

        # Python 3.8+ on Windows needs explicit add_dll_directory
        if hasattr(os, 'add_dll_directory'):
            for p in dll_paths:
                if p.exists():
                    try:
                        os.add_dll_directory(str(p))
                    except Exception:
                        pass

if __name__ == '__main__':
    # CRITICAL: Set up DLL paths BEFORE any imports that might load torch
    setup_dll_paths()

    # Now import and run the main application
    # Use exec to avoid import-time torch loading issues
    import importlib.util

    # Find the main script
    if getattr(sys, 'frozen', False):
        main_script = Path(sys._MEIPASS) / 'shaft_segmentation_gui_main.py'
    else:
        main_script = Path(__file__).parent / 'shaft_segmentation_gui.py'

    if main_script.exists():
        spec = importlib.util.spec_from_file_location("main", main_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        # Fallback: direct import
        from shaft_segmentation_gui import main
        main()
