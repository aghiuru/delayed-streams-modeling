"""
Setup script for building the macOS subtitle app with py2app.

Usage:
    python setup.py py2app
"""

from setuptools import setup

APP = ["macos_subtitle_app.py"]
DATA_FILES = []
OPTIONS = {
    "argv_emulation": False,
    "packages": [
        "subtitle_app",
        "rumps",
        "PySide6",
        "mlx",
        "moshi_mlx",
        "rustymimi",
        "sentencepiece",
        "sounddevice",
        "huggingface_hub",
        "requests",
        "numpy",
    ],
    "iconfile": None,  # Add path to .icns file if you have one
    "plist": {
        "CFBundleName": "Subtitle App",
        "CFBundleDisplayName": "Subtitle App",
        "CFBundleIdentifier": "com.kyutai.subtitleapp",
        "CFBundleVersion": "0.1.0",
        "CFBundleShortVersionString": "0.1.0",
        "NSHighResolutionCapable": True,
        "LSUIElement": True,  # Hide from dock (menu bar app only)
        "NSMicrophoneUsageDescription": "This app needs microphone access for real-time transcription.",
    },
    "includes": [
        "shiboken6",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
    ],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
