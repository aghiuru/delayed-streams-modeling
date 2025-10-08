# macOS Subtitle App

A real-time speech-to-text transcription application with translation overlay for macOS.

## Features

- 🎤 **Real-time transcription** using MLX-optimized Moshi models
- 🌍 **Live translation** via Ollama (supports any language)
- 💬 **Overlay subtitles** displayed over any application
- ⚙️ **Configurable** timing, appearance, and behavior
- 🍎 **Native macOS** menu bar integration

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3)
- **Python 3.12**
- **Ollama** running locally for translation (install from [ollama.ai](https://ollama.ai))

## Installation

### 1. Install Dependencies

Using Poetry (recommended):

```bash
# Install with GUI dependencies
poetry install --extras gui
```

Or using pip:

```bash
pip install -e ".[gui]"
```

### 2. Install Ollama Model

The app uses Ollama for translation. Install and pull the translation model:

```bash
# Install Ollama from https://ollama.ai
# Then pull the translation model:
ollama pull zongwei/gemma3-translator:4b
```

### 3. Run the App

```bash
poetry run python macos_subtitle_app.py
```

The app will appear in your menu bar (look for the "Subtitles" icon).

## Usage

### Quick Start

1. Click the **Subtitles** menu bar icon
2. Select **"Start Recording"**
3. Wait for models to load (first time takes ~30 seconds)
4. Speak! Subtitles will appear at the bottom of your screen

### Configuration

Click **"Settings..."** in the menu to configure:

- **Audio Tab**: Select input device
- **Translation Tab**: Configure Ollama host, model, target language
- **Subtitle Timing Tab**: Adjust duration, word limits
- **Appearance Tab**: Font, colors, position
- **Model Tab**: HuggingFace model, cache directory

Settings are saved to `~/.subtitle_app/config.json`

### Menu Options

- **Start/Stop Recording**: Toggle transcription
- **Show/Hide Subtitles**: Toggle subtitle window visibility
- **Settings...**: Open configuration dialog
- **Quit**: Exit the application

## Building a Standalone App

**⚠️ Note**: Building a standalone app is complex due to large ML dependencies (~2GB models). For most users, it's recommended to run the app directly with Poetry.

### Option 1: Run Directly (Recommended)

```bash
# Install dependencies
poetry install --extras gui

# Run the app
poetry run python macos_subtitle_app.py --debug
```

### Option 2: Build Standalone App (Advanced)

py2app support for this project is experimental due to:
- Large ML model files (~2GB)
- Binary extensions (MLX, rustymimi)
- Complex dependency tree

If you want to try building:

```bash
# Install build dependencies
poetry install --extras gui

# Attempt to build (may require manual intervention)
poetry run python setup.py py2app
```

**Known Issues**:
- py2app may fail to bundle MLX and other binary dependencies
- Models are not included - app would need to download on first run
- App bundle would be very large (2GB+)

**Recommended Alternative**: Create a simple launcher script that activates the poetry environment and runs the app.

## Configuration

### Default Settings

The app uses sensible defaults:

- **Model**: `kyutai/stt-1b-en_fr-mlx`
- **Target Language**: Spanish
- **Subtitle Timing**: 1.5-5 seconds per segment
- **Display**: Bottom center, 2 lines max, 7 words per line
- **Font**: Arial, 32pt, white text on black background

### Custom Settings

Edit `~/.subtitle_app/config.json` or use the Settings dialog.

Example configuration:

```json
{
  "target_language": "French",
  "font_size": 40,
  "window_position": "top",
  "background_opacity": 0.8
}
```

## Troubleshooting

### "Failed to load models"

- Ensure you have enough free RAM (~4GB)
- Check internet connection for first-time model download
- Models are cached in `~/.cache/huggingface` by default

### Translation shows original text

- Ensure Ollama is running: `ollama serve`
- Check the model is installed: `ollama list`
- Verify Ollama host in Settings (default: `http://localhost:11434`)

### No audio input

- Check microphone permissions in System Settings → Privacy & Security → Microphone
- Select correct audio device in Settings → Audio tab

### Subtitles not appearing

- Click "Show Subtitles" in menu bar
- Check that recording is started
- Try speaking clearly and loudly

### App is slow

- First transcription is slow due to model loading
- Subsequent transcriptions are much faster
- Consider using quantized models (`.q4` or `.q8`) for better performance

## Architecture

The app consists of several components:

```
macos_subtitle_app.py          # Main entry point
subtitle_app/
├── __init__.py                # Package init
├── config.py                  # Configuration management
├── menu_bar.py                # rumps menu bar app
├── subtitle_window.py         # PySide6 frameless subtitle display
├── config_window.py           # PySide6 settings dialog
└── transcription_engine.py    # Core STT and translation logic
```

### Key Technologies

- **rumps**: macOS menu bar framework
- **PySide6**: Qt bindings for GUI windows
- **MLX**: Apple Silicon ML framework
- **moshi_mlx**: Speech-to-text model
- **Ollama**: Local LLM for translation

## Development

### Running in Development Mode

```bash
python macos_subtitle_app.py
```

### Project Structure

- Original CLI script: `scripts/stt_from_mic_mlx_subtitles.py`
- GUI app: `macos_subtitle_app.py` + `subtitle_app/`
- Configuration: Uses Poetry for dependency management

### Adding Features

The modular design makes it easy to extend:

- Add menu items in `menu_bar.py`
- Add settings in `config.py` and `config_window.py`
- Modify subtitle appearance in `subtitle_window.py`
- Change transcription logic in `transcription_engine.py`

## License

MIT License - See main project README for details.

## Credits

Based on the Kyutai Labs delayed streams modeling project.
