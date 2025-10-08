"""
Subtitle display window for the macOS app.

A frameless, transparent window that displays subtitles over other applications.
"""

import logging
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout

logger = logging.getLogger(__name__)


class SubtitleWindow(QWidget):
    """
    A frameless window that displays subtitles.

    Features:
    - Transparent background
    - Always on top
    - No window controls
    - Configurable appearance
    """

    _instance_count = 0  # Class variable to track instances

    def __init__(self, config):
        """Initialize the subtitle window with configuration."""
        super().__init__()
        SubtitleWindow._instance_count += 1
        self.instance_id = SubtitleWindow._instance_count
        logger.info(f"Initializing subtitle window (instance #{self.instance_id})")
        logger.warning(f"Total SubtitleWindow instances created: {SubtitleWindow._instance_count}")
        self.config = config
        self.current_lines = []

        # For dragging the window
        self._drag_position = None

        self.setup_ui()
        self.apply_config()
        logger.info(f"Subtitle window #{self.instance_id} initialized")

    def setup_ui(self):
        """Set up the user interface."""
        # Window flags for frameless, transparent, always-on-top window
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool  # Don't show in dock
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(layout)

        # Create label for subtitle text
        self.subtitle_label = QLabel("")
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setWordWrap(True)

        layout.addWidget(self.subtitle_label)

        # Start hidden
        self.hide()

    def apply_config(self):
        """Apply configuration to window appearance."""
        logger.debug("Applying window configuration")
        # Set window geometry
        if self.config.window_position == "bottom":
            # Position at bottom center of screen
            from PySide6.QtWidgets import QApplication

            screen = QApplication.primaryScreen().geometry()
            x = (screen.width() - self.config.window_width) // 2
            y = screen.height() - self.config.window_height - 50
            self.setGeometry(x, y, self.config.window_width, self.config.window_height)
            logger.debug(f"Window positioned at bottom: ({x}, {y}, {self.config.window_width}, {self.config.window_height})")
        elif self.config.window_position == "top":
            # Position at top center of screen
            from PySide6.QtWidgets import QApplication

            screen = QApplication.primaryScreen().geometry()
            x = (screen.width() - self.config.window_width) // 2
            y = 50
            self.setGeometry(x, y, self.config.window_width, self.config.window_height)
            logger.debug(f"Window positioned at top: ({x}, {y}, {self.config.window_width}, {self.config.window_height})")
        else:
            # Use custom position
            self.setGeometry(
                self.config.window_x,
                self.config.window_y,
                self.config.window_width,
                self.config.window_height,
            )
            logger.debug(f"Window positioned at custom: ({self.config.window_x}, {self.config.window_y}, {self.config.window_width}, {self.config.window_height})")

        # Set font
        font = QFont(self.config.font_family, self.config.font_size)
        font.setBold(True)
        self.subtitle_label.setFont(font)

        # Set colors with opacity
        bg_color = QColor(self.config.background_color)
        bg_color.setAlphaF(self.config.background_opacity)

        fg_color = QColor(self.config.font_color)

        # Apply stylesheet
        self.subtitle_label.setStyleSheet(
            f"""
            QLabel {{
                color: {fg_color.name()};
                background-color: rgba({bg_color.red()}, {bg_color.green()}, {bg_color.blue()}, {int(bg_color.alphaF() * 255)});
                padding: 20px;
                border-radius: 10px;
            }}
            """
        )

    def update_subtitle(self, lines):
        """
        Update the subtitle text with new lines.

        Args:
            lines: List of strings, one per line
        """
        logger.info(f"[Window #{self.instance_id}] update_subtitle called with {len(lines) if lines else 0} lines")
        if lines:
            logger.debug(f"[Window #{self.instance_id}] Subtitle content: {lines}")

        if not lines:
            logger.debug(f"[Window #{self.instance_id}] No lines to display, clearing subtitle")
            self.subtitle_label.setText("")
            return

        # Clear old text first, then set new text
        logger.debug(f"[Window #{self.instance_id}] Clearing old text before update")
        old_text = self.subtitle_label.text()
        logger.debug(f"[Window #{self.instance_id}] Old text was: '{old_text}'")
        self.subtitle_label.clear()  # Explicitly clear first
        self.subtitle_label.repaint()  # Force repaint after clear

        # Join lines with newline
        text = "\n".join(lines)
        logger.info(f"[Window #{self.instance_id}] Setting subtitle text: '{text}'")
        self.subtitle_label.setText(text)
        self.current_lines = lines
        self.subtitle_label.repaint()  # Force repaint after setting text

        # Log current label text to verify
        current_text = self.subtitle_label.text()
        logger.debug(f"[Window #{self.instance_id}] Label now contains: '{current_text}'")

        # Make sure window is visible
        if not self.isVisible():
            logger.info(f"[Window #{self.instance_id}] Window was hidden, showing it now")
            self.show()
        else:
            logger.debug(f"[Window #{self.instance_id}] Window already visible")

    def clear_subtitle(self):
        """Clear the current subtitle."""
        logger.debug("Clearing subtitle")
        self.subtitle_label.setText("")
        self.current_lines = []

    def show(self):
        """Override show to add logging."""
        logger.info(f"[Window #{self.instance_id}] Showing subtitle window")
        super().show()
        logger.info(f"[Window #{self.instance_id}] Window visibility: {self.isVisible()}")

    def hide(self):
        """Override hide to add logging."""
        logger.info(f"[Window #{self.instance_id}] Hiding subtitle window")
        super().hide()

    def update_config(self, config):
        """Update configuration and reapply settings."""
        logger.info("Updating subtitle window configuration")
        self.config = config
        self.apply_config()

    def mousePressEvent(self, event):
        """Handle mouse press to start dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            logger.debug(f"[Window #{self.instance_id}] Mouse pressed, starting drag")
            event.accept()

    def mouseMoveEvent(self, event):
        """Handle mouse move to drag the window."""
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_position is not None:
            new_pos = event.globalPosition().toPoint() - self._drag_position
            self.move(new_pos)
            logger.debug(f"[Window #{self.instance_id}] Window moved to: {new_pos}")
            event.accept()

    def mouseReleaseEvent(self, event):
        """Handle mouse release to stop dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = None
            logger.debug(f"[Window #{self.instance_id}] Mouse released, stopped dragging")
            event.accept()
