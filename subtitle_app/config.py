"""
Configuration management for the subtitle app.

Handles loading, saving, and accessing user settings.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class SubtitleConfig:
    """Application configuration settings."""

    # Debug settings
    debug_mode: bool = False

    # Model settings
    hf_repo: str = "kyutai/stt-1b-en_fr-mlx"
    cache_dir: Optional[str] = None
    max_steps: int = 4096

    # Audio settings
    audio_device: Optional[str] = None  # None = default device

    # Translation settings
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "zongwei/gemma3-translator:4b"
    target_language: str = "Romanian"
    translation_timeout: float = 3.0

    # Subtitle timing
    min_duration: float = 1.5
    max_duration: float = 5.0
    soft_max_words: int = 10

    # Subtitle display
    max_lines: int = 2
    words_per_line: int = 7

    # Window appearance
    font_family: str = "Arial"
    font_size: int = 32
    font_color: str = "#FFFFFF"
    background_color: str = "#000000"
    background_opacity: float = 0.7

    # Window position
    window_x: int = 100
    window_y: int = 100
    window_width: int = 800
    window_height: int = 150
    window_position: str = "bottom"  # "top" or "bottom"

    @classmethod
    def get_config_path(cls) -> Path:
        """Get the path to the config file."""
        config_dir = Path.home() / ".subtitle_app"
        config_dir.mkdir(exist_ok=True)
        return config_dir / "config.json"

    @classmethod
    def load(cls) -> "SubtitleConfig":
        """Load configuration from file, or return defaults if not found."""
        config_path = cls.get_config_path()
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    data = json.load(f)
                return cls(**data)
            except Exception as e:
                print(f"Error loading config: {e}. Using defaults.")
                return cls()
        return cls()

    def save(self) -> None:
        """Save configuration to file."""
        config_path = self.get_config_path()
        try:
            with open(config_path, "w") as f:
                json.dump(asdict(self), f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)
