# -*- mode: python ; coding: utf-8 -*-
import PyInstaller.config

PyInstaller.config.CONF['distpath'] = 'dist'
PyInstaller.config.CONF['workpath'] = 'build'

a = Analysis(
    ['match_faces_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('face_detection_model', 'face_detection_model'),
        ('openface_nn4.small2.v1.t7', '.'),
        ('images', 'images'),
    ],
    hiddenimports=['cv2'],
    hookspath=[],
    runtime_hooks=[],
    excludedimports=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='FaceRecognition',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
)
