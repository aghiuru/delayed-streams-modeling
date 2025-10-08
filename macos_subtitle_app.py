#!/usr/bin/env python3
"""
macOS Subtitle App - Real-time transcription with translation overlay.

This is the main entry point for the application. It initializes both
PySide6 (for the subtitle and settings windows) and rumps (for the menu bar).
"""

import sys
import argparse
import logging
from pathlib import Path
from PySide6.QtWidgets import QApplication
from subtitle_app.menu_bar import SubtitleMenuBar


def setup_logging(debug_mode: bool):
    """Set up logging to console and file."""
    # Create log directory
    log_dir = Path.home() / ".subtitle_app"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "debug.log"

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    # File handler (always log to file)
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (only if debug mode)
    if debug_mode:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return log_file


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='macOS Subtitle App')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging to console and file'
    )
    args = parser.parse_args()

    # Set up logging
    log_file = setup_logging(args.debug)

    if args.debug:
        print(f"Debug mode enabled. Logging to console and {log_file}")

    logging.info("=" * 80)
    logging.info("Starting macOS Subtitle App")
    logging.info(f"Debug mode: {args.debug}")
    logging.info("=" * 80)

    # Create Qt application
    qt_app = QApplication(sys.argv)
    qt_app.setQuitOnLastWindowClosed(False)  # Keep running when windows close

    # Create menu bar app
    menu_app = SubtitleMenuBar(qt_app, debug_mode=args.debug)

    # Run the menu bar app (this will also process Qt events)
    menu_app.run()


if __name__ == "__main__":
    main()
