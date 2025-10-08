"""
Configuration window for the subtitle app.

Provides a GUI for modifying application settings.
"""

import sounddevice as sd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QPushButton,
    QGroupBox,
    QColorDialog,
    QTabWidget,
    QWidget,
)
from PySide6.QtGui import QColor


class ConfigWindow(QDialog):
    """Configuration dialog for the subtitle app."""

    def __init__(self, config, parent=None):
        """Initialize the configuration window."""
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Subtitle App Settings")
        self.setMinimumWidth(600)
        self.setup_ui()
        self.load_values()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create tabs
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Audio tab
        audio_tab = self.create_audio_tab()
        tabs.addTab(audio_tab, "Audio")

        # Translation tab
        translation_tab = self.create_translation_tab()
        tabs.addTab(translation_tab, "Translation")

        # Subtitle Timing tab
        timing_tab = self.create_timing_tab()
        tabs.addTab(timing_tab, "Subtitle Timing")

        # Appearance tab
        appearance_tab = self.create_appearance_tab()
        tabs.addTab(appearance_tab, "Appearance")

        # Model tab
        model_tab = self.create_model_tab()
        tabs.addTab(model_tab, "Model")

        # Buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_config)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    def create_audio_tab(self):
        """Create the audio settings tab."""
        widget = QWidget()
        layout = QFormLayout()
        widget.setLayout(layout)

        # Audio device selector
        self.audio_device_combo = QComboBox()
        self.audio_device_combo.addItem("Default", None)

        # Get available audio devices
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device["max_input_channels"] > 0:
                    self.audio_device_combo.addItem(device["name"], i)
        except Exception as e:
            print(f"Error querying audio devices: {e}")

        layout.addRow("Audio Input Device:", self.audio_device_combo)

        return widget

    def create_translation_tab(self):
        """Create the translation settings tab."""
        widget = QWidget()
        layout = QFormLayout()
        widget.setLayout(layout)

        self.ollama_host_edit = QLineEdit()
        layout.addRow("Ollama Host:", self.ollama_host_edit)

        self.ollama_model_edit = QLineEdit()
        layout.addRow("Ollama Model:", self.ollama_model_edit)

        self.target_language_edit = QLineEdit()
        layout.addRow("Target Language:", self.target_language_edit)

        self.translation_timeout_spin = QDoubleSpinBox()
        self.translation_timeout_spin.setMinimum(0.5)
        self.translation_timeout_spin.setMaximum(30.0)
        self.translation_timeout_spin.setSingleStep(0.5)
        self.translation_timeout_spin.setSuffix(" seconds")
        layout.addRow("Translation Timeout:", self.translation_timeout_spin)

        return widget

    def create_timing_tab(self):
        """Create the subtitle timing settings tab."""
        widget = QWidget()
        layout = QFormLayout()
        widget.setLayout(layout)

        self.min_duration_spin = QDoubleSpinBox()
        self.min_duration_spin.setMinimum(0.5)
        self.min_duration_spin.setMaximum(10.0)
        self.min_duration_spin.setSingleStep(0.1)
        self.min_duration_spin.setSuffix(" seconds")
        layout.addRow("Min Duration:", self.min_duration_spin)

        self.max_duration_spin = QDoubleSpinBox()
        self.max_duration_spin.setMinimum(1.0)
        self.max_duration_spin.setMaximum(20.0)
        self.max_duration_spin.setSingleStep(0.5)
        self.max_duration_spin.setSuffix(" seconds")
        layout.addRow("Max Duration:", self.max_duration_spin)

        self.soft_max_words_spin = QSpinBox()
        self.soft_max_words_spin.setMinimum(5)
        self.soft_max_words_spin.setMaximum(30)
        layout.addRow("Soft Max Words:", self.soft_max_words_spin)

        self.max_lines_spin = QSpinBox()
        self.max_lines_spin.setMinimum(1)
        self.max_lines_spin.setMaximum(3)
        layout.addRow("Max Lines:", self.max_lines_spin)

        self.words_per_line_spin = QSpinBox()
        self.words_per_line_spin.setMinimum(3)
        self.words_per_line_spin.setMaximum(15)
        layout.addRow("Words Per Line:", self.words_per_line_spin)

        return widget

    def create_appearance_tab(self):
        """Create the appearance settings tab."""
        widget = QWidget()
        layout = QFormLayout()
        widget.setLayout(layout)

        # Font settings
        self.font_family_edit = QLineEdit()
        layout.addRow("Font Family:", self.font_family_edit)

        self.font_size_spin = QSpinBox()
        self.font_size_spin.setMinimum(12)
        self.font_size_spin.setMaximum(72)
        layout.addRow("Font Size:", self.font_size_spin)

        # Color settings
        color_layout = QHBoxLayout()
        self.font_color_edit = QLineEdit()
        self.font_color_edit.setReadOnly(True)
        font_color_button = QPushButton("Choose...")
        font_color_button.clicked.connect(self.choose_font_color)
        color_layout.addWidget(self.font_color_edit)
        color_layout.addWidget(font_color_button)
        layout.addRow("Font Color:", color_layout)

        bg_color_layout = QHBoxLayout()
        self.bg_color_edit = QLineEdit()
        self.bg_color_edit.setReadOnly(True)
        bg_color_button = QPushButton("Choose...")
        bg_color_button.clicked.connect(self.choose_bg_color)
        bg_color_layout.addWidget(self.bg_color_edit)
        bg_color_layout.addWidget(bg_color_button)
        layout.addRow("Background Color:", bg_color_layout)

        self.bg_opacity_spin = QDoubleSpinBox()
        self.bg_opacity_spin.setMinimum(0.0)
        self.bg_opacity_spin.setMaximum(1.0)
        self.bg_opacity_spin.setSingleStep(0.1)
        layout.addRow("Background Opacity:", self.bg_opacity_spin)

        # Position settings
        self.window_position_combo = QComboBox()
        self.window_position_combo.addItems(["top", "bottom", "custom"])
        layout.addRow("Window Position:", self.window_position_combo)

        self.window_width_spin = QSpinBox()
        self.window_width_spin.setMinimum(200)
        self.window_width_spin.setMaximum(3000)
        self.window_width_spin.setSingleStep(50)
        layout.addRow("Window Width:", self.window_width_spin)

        self.window_height_spin = QSpinBox()
        self.window_height_spin.setMinimum(50)
        self.window_height_spin.setMaximum(1000)
        self.window_height_spin.setSingleStep(10)
        layout.addRow("Window Height:", self.window_height_spin)

        return widget

    def create_model_tab(self):
        """Create the model settings tab."""
        widget = QWidget()
        layout = QFormLayout()
        widget.setLayout(layout)

        self.hf_repo_edit = QLineEdit()
        layout.addRow("HuggingFace Repo:", self.hf_repo_edit)

        self.cache_dir_edit = QLineEdit()
        layout.addRow("Cache Directory:", self.cache_dir_edit)

        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setMinimum(512)
        self.max_steps_spin.setMaximum(8192)
        self.max_steps_spin.setSingleStep(512)
        layout.addRow("Max Steps:", self.max_steps_spin)

        return widget

    def load_values(self):
        """Load values from config into the form."""
        # Audio
        if self.config.audio_device is not None:
            index = self.audio_device_combo.findData(self.config.audio_device)
            if index >= 0:
                self.audio_device_combo.setCurrentIndex(index)

        # Translation
        self.ollama_host_edit.setText(self.config.ollama_host)
        self.ollama_model_edit.setText(self.config.ollama_model)
        self.target_language_edit.setText(self.config.target_language)
        self.translation_timeout_spin.setValue(self.config.translation_timeout)

        # Timing
        self.min_duration_spin.setValue(self.config.min_duration)
        self.max_duration_spin.setValue(self.config.max_duration)
        self.soft_max_words_spin.setValue(self.config.soft_max_words)
        self.max_lines_spin.setValue(self.config.max_lines)
        self.words_per_line_spin.setValue(self.config.words_per_line)

        # Appearance
        self.font_family_edit.setText(self.config.font_family)
        self.font_size_spin.setValue(self.config.font_size)
        self.font_color_edit.setText(self.config.font_color)
        self.bg_color_edit.setText(self.config.background_color)
        self.bg_opacity_spin.setValue(self.config.background_opacity)
        self.window_position_combo.setCurrentText(self.config.window_position)
        self.window_width_spin.setValue(self.config.window_width)
        self.window_height_spin.setValue(self.config.window_height)

        # Model
        self.hf_repo_edit.setText(self.config.hf_repo)
        if self.config.cache_dir:
            self.cache_dir_edit.setText(self.config.cache_dir)
        self.max_steps_spin.setValue(self.config.max_steps)

    def choose_font_color(self):
        """Open color dialog for font color."""
        color = QColorDialog.getColor(
            QColor(self.config.font_color), self, "Choose Font Color"
        )
        if color.isValid():
            self.font_color_edit.setText(color.name())

    def choose_bg_color(self):
        """Open color dialog for background color."""
        color = QColorDialog.getColor(
            QColor(self.config.background_color), self, "Choose Background Color"
        )
        if color.isValid():
            self.bg_color_edit.setText(color.name())

    def save_config(self):
        """Save the configuration and close the dialog."""
        # Audio
        self.config.audio_device = self.audio_device_combo.currentData()

        # Translation
        self.config.ollama_host = self.ollama_host_edit.text()
        self.config.ollama_model = self.ollama_model_edit.text()
        self.config.target_language = self.target_language_edit.text()
        self.config.translation_timeout = self.translation_timeout_spin.value()

        # Timing
        self.config.min_duration = self.min_duration_spin.value()
        self.config.max_duration = self.max_duration_spin.value()
        self.config.soft_max_words = self.soft_max_words_spin.value()
        self.config.max_lines = self.max_lines_spin.value()
        self.config.words_per_line = self.words_per_line_spin.value()

        # Appearance
        self.config.font_family = self.font_family_edit.text()
        self.config.font_size = self.font_size_spin.value()
        self.config.font_color = self.font_color_edit.text()
        self.config.background_color = self.bg_color_edit.text()
        self.config.background_opacity = self.bg_opacity_spin.value()
        self.config.window_position = self.window_position_combo.currentText()
        self.config.window_width = self.window_width_spin.value()
        self.config.window_height = self.window_height_spin.value()

        # Model
        self.config.hf_repo = self.hf_repo_edit.text()
        cache_dir = self.cache_dir_edit.text().strip()
        self.config.cache_dir = cache_dir if cache_dir else None
        self.config.max_steps = self.max_steps_spin.value()

        # Save to file
        self.config.save()

        self.accept()
