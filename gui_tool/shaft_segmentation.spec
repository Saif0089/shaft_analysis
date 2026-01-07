# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Shaft Segmentation Tool

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all torch submodules
torch_hidden_imports = collect_submodules('torch')

# Additional hidden imports
hidden_imports = [
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.utils.data',
    'numpy',
    'laspy',
    'plyfile',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends.backend_agg',
    'PIL',
    'PyQt5',
    'PyQt5.QtWidgets',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.sip',
] + torch_hidden_imports

# Collect data files
datas = [
    ('checkpoints/*.pth', 'checkpoints'),
    ('models/*.py', 'models'),
]

# Try to collect torch data files
try:
    datas += collect_data_files('torch')
except:
    pass

a = Analysis(
    ['shaft_segmentation_gui.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',  # We use PyQt5
        'test',
        'tests',
        'unittest',
        'pydoc',
        'doctest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ShaftSegmentation',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one
)
