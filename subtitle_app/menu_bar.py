"""
Menu bar application for the subtitle app.

Uses rumps to create a macOS menu bar app that controls the transcription
and subtitle display.
"""

import logging
import rumps
from PySide6.QtCore import QTimer, QObject, Signal, Slot
from .config import SubtitleConfig
from .config_window import ConfigWindow
from .subtitle_window import SubtitleWindow
from .transcription_engine import TranscriptionEngine, translate_with_ollama

logger = logging.getLogger(__name__)


class SubtitleSignalHandler(QObject):
    """Qt signal handler for thread-safe subtitle updates."""
    subtitle_ready = Signal(object)  # Signal emits SubtitleSegment


class SubtitleMenuBar(rumps.App):
    """Menu bar app for controlling subtitle transcription."""

    def __init__(self, qt_app, debug_mode=False):
        """Initialize the menu bar app."""
        super().__init__("Subtitles", icon=None, quit_button=None)
        self.qt_app = qt_app

        logger.info("Initializing Subtitle Menu Bar")

        # Load configuration
        self.config = SubtitleConfig.load()
        self.config.debug_mode = debug_mode

        logger.info(f"Configuration loaded: {self.config.hf_repo}")
        logger.info(f"Target language: {self.config.target_language}")

        # Test translation connection
        self._test_translation()

        # Initialize components (subtitle_window created lazily)
        self._subtitle_window = None

        # Create signal handler for thread-safe subtitle updates
        self.signal_handler = SubtitleSignalHandler()
        self.signal_handler.subtitle_ready.connect(self._handle_subtitle_update)

        self.transcription_engine = TranscriptionEngine(self.config)
        self.transcription_engine.on_subtitle_callback = self.on_new_subtitle

        # Setup menu
        self.menu = [
            rumps.MenuItem("Start Recording", callback=self.toggle_recording),
            rumps.separator,
            rumps.MenuItem("Show Subtitles", callback=self.toggle_subtitle_window),
            rumps.MenuItem("Settings...", callback=self.show_settings),
            rumps.separator,
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]

        self.is_recording = False
        self.subtitle_window_visible = False
        self.update_menu_state()

        logger.info("Menu bar initialized successfully")

    def _test_translation(self):
        """Test the translation connection like the original script."""
        logger.info("Testing translation connection...")
        logger.info(f"  Ollama host: {self.config.ollama_host}")
        logger.info(f"  Model: {self.config.ollama_model}")
        logger.info(f"  Target language: {self.config.target_language}")
        logger.info(f"  Timeout: {self.config.translation_timeout}s")

        test_text = "Hello world"
        logger.info(f"  Testing with: '{test_text}'")

        test_result = translate_with_ollama(
            test_text,
            target_language=self.config.target_language,
            ollama_host=self.config.ollama_host,
            model=self.config.ollama_model,
            timeout=self.config.translation_timeout,
        )

        if test_result == test_text:
            logger.warning("  ⚠️  Translation failed - will show original text only")
        else:
            logger.info(f"  ✅ Translation works: '{test_result}'")

    @property
    def subtitle_window(self):
        """Lazy-load the subtitle window on first access."""
        if self._subtitle_window is None:
            logger.warning("subtitle_window property: Creating NEW subtitle window")
            self._subtitle_window = SubtitleWindow(self.config)
        else:
            logger.debug(f"subtitle_window property: Returning existing window #{self._subtitle_window.instance_id}")
        return self._subtitle_window

    def update_menu_state(self):
        """Update menu item states based on current state."""
        if self.is_recording:
            self.menu["Start Recording"].title = "Stop Recording"
        else:
            self.menu["Start Recording"].title = "Start Recording"

        if self.subtitle_window_visible:
            self.menu["Show Subtitles"].title = "Hide Subtitles"
        else:
            self.menu["Show Subtitles"].title = "Show Subtitles"

    def toggle_recording(self, sender):
        """Toggle recording on/off."""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """Start the transcription process."""
        if not self.is_recording:
            logger.info("User requested to start recording")
            # Show loading notification
            rumps.notification(
                "Subtitle App",
                "Starting...",
                "Loading models and starting transcription. This may take a moment.",
            )

            # Start in a separate thread to avoid blocking
            def load_and_start():
                logger.info("Loading models and starting recording...")
                success = self.transcription_engine.load_models()
                if success:
                    self.transcription_engine.start_recording()
                    self.is_recording = True
                    self.update_menu_state()
                    rumps.notification(
                        "Subtitle App", "Recording Started", "Speak to see subtitles."
                    )
                    logger.info("Recording started successfully")
                    # Show subtitle window if not visible (must be on main thread)
                    logger.debug(f"Checking if should show window: subtitle_window_visible={self.subtitle_window_visible}")
                    if not self.subtitle_window_visible:
                        def show_window():
                            logger.info("show_window() executing - showing subtitle window")
                            self.subtitle_window.show()
                            self.subtitle_window_visible = True
                            logger.info(f"subtitle_window_visible flag now set to: {self.subtitle_window_visible}")
                            self.update_menu_state()
                        QTimer.singleShot(0, show_window)
                    else:
                        logger.debug("Window already marked as visible")
                else:
                    logger.error("Failed to load models")
                    rumps.alert(
                        "Error",
                        "Failed to load models. Check console for details.",
                    )

            import threading

            threading.Thread(target=load_and_start, daemon=True).start()

    def stop_recording(self):
        """Stop the transcription process."""
        if self.is_recording:
            logger.info("User requested to stop recording")
            self.transcription_engine.stop_recording()
            self.is_recording = False
            # Clear subtitle on main thread
            QTimer.singleShot(0, self.subtitle_window.clear_subtitle)
            self.update_menu_state()
            rumps.notification("Subtitle App", "Recording Stopped", "")
            logger.info("Recording stopped by user")

    def toggle_subtitle_window(self, sender):
        """Toggle subtitle window visibility."""
        def toggle():
            if self.subtitle_window_visible:
                logger.info("Hiding subtitle window")
                self.subtitle_window.hide()
                self.subtitle_window_visible = False
            else:
                logger.info("Showing subtitle window")
                self.subtitle_window.show()
                self.subtitle_window_visible = True
            self.update_menu_state()

        # Must run on main thread
        QTimer.singleShot(0, toggle)

    def show_settings(self, sender):
        """Show the settings dialog."""
        logger.info("Opening settings dialog")
        # Need to run dialog in Qt event loop
        def open_dialog():
            dialog = ConfigWindow(self.config)
            if dialog.exec():
                logger.info("Settings saved")
                # Configuration was saved
                # Update subtitle window config if it exists
                if self._subtitle_window is not None:
                    self._subtitle_window.update_config(self.config)
                # If recording, need to restart with new config
                if self.is_recording:
                    logger.warning("Recording active - restart needed for changes to take effect")
                    rumps.alert(
                        "Configuration Updated",
                        "Stop and restart recording for changes to take effect.",
                    )
            else:
                logger.info("Settings dialog cancelled")

        # Schedule dialog to open in Qt event loop
        QTimer.singleShot(0, open_dialog)

    def on_new_subtitle(self, segment):
        """Callback when a new subtitle segment is ready (called from background thread)."""
        logger.info(f"New subtitle received: '{segment.translated_text}'")
        logger.debug(f"Emitting subtitle_ready signal from thread")
        # Emit signal for thread-safe communication to Qt main thread
        self.signal_handler.subtitle_ready.emit(segment)

    @Slot(object)
    def _handle_subtitle_update(self, segment):
        """Handle subtitle update on Qt main thread."""
        logger.info(f"_handle_subtitle_update called on main thread")
        logger.debug(f"subtitle_window_visible flag: {self.subtitle_window_visible}")

        if self.subtitle_window_visible:
            logger.debug(f"Updating subtitle display with text: '{segment.lines}'")
            self.subtitle_window.update_subtitle(segment.lines)
        else:
            logger.warning("Subtitle window not visible flag is False, skipping update")

    def quit_app(self, sender):
        """Quit the application."""
        logger.info("Quitting application")
        if self.is_recording:
            logger.info("Stopping recording before quit")
            self.stop_recording()
        logger.info("Application shutting down")
        rumps.quit_application()
