# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "moshi_mlx==0.2.12",
#     "numpy",
#     "requests",
#     "rustymimi",
#     "sentencepiece",
#     "sounddevice",
# ]
# ///

import argparse
import json
import os
import queue
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import mlx.core as mx
import mlx.nn as nn
import requests
import rustymimi
import sentencepiece
import sounddevice as sd
from huggingface_hub import hf_hub_download
from moshi_mlx import models, utils


@dataclass
class SubtitleSegment:
    """Represents a completed subtitle segment."""
    text: str
    translated_text: str
    lines: List[str]
    start_time: float
    end_time: float
    word_count: int
    char_count: int


def translate_with_ollama(
    text: str,
    target_language: str,
    ollama_host: str = "http://localhost:11434",
    model: str = "zongwei/gemma3-translator:4b",
    timeout: float = 3.0,
) -> str:
    """
    Translate text using Ollama API.
    Returns original text if translation fails.
    """
    if not text.strip():
        return text

    # Craft translation prompt
    prompt = f"""Translate to {target_language}. Output only the translation, no explanations.

{text}"""

    try:
        response = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=timeout,
        )

        if response.status_code == 200:
            result = response.json()
            translated = result.get("response", "").strip()
            return translated if translated else text
        else:
            print(f"\nWarning: Translation API returned status {response.status_code}")
            return text

    except requests.exceptions.Timeout:
        print("\nWarning: Translation timeout, showing original text")
        return text
    except requests.exceptions.ConnectionError:
        print("\nWarning: Cannot connect to Ollama, showing original text")
        return text
    except Exception as e:
        print(f"\nWarning: Translation error ({e}), showing original text")
        return text


class SubtitleBatcher:
    """Batches words into subtitle segments with real-time constraints."""

    def __init__(
        self,
        min_duration: float = 1.5,
        max_duration: float = 5.0,
        max_lines: int = 2,
        words_per_line: int = 7,
    ):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_lines = max_lines
        self.words_per_line = words_per_line
        self.max_words = max_lines * words_per_line

        self.current_words: List[str] = []
        self.segment_start_time: float = 0.0
        self.last_flush_time: float = 0.0
        self.completed_segments: List[SubtitleSegment] = []

    @staticmethod
    def break_into_lines(text: str, max_lines: int, words_per_line: int) -> List[str]:
        """
        Break text into multiple lines based on word count.
        Distributes words evenly across lines.
        """
        if not text:
            return []

        words = text.strip().split()
        if not words:
            return []

        # If text fits on one line, return it
        if len(words) <= words_per_line:
            return [text]

        lines = []
        current_line_words = []

        for i, word in enumerate(words):
            current_line_words.append(word)

            # Check if current line is full
            if len(current_line_words) >= words_per_line:
                # Line is full, add it
                lines.append(" ".join(current_line_words))
                current_line_words = []

                # Stop if we've reached max lines
                if len(lines) >= max_lines:
                    # Add any remaining words to last line (overflow)
                    remaining_words = words[i + 1:]
                    if remaining_words:
                        lines[-1] = lines[-1] + " " + " ".join(remaining_words)
                    return lines

        # Add any remaining words as the last line
        if current_line_words:
            lines.append(" ".join(current_line_words))

        return lines

    def add_word(self, word: str, current_time: float):
        """Add a word to the current batch."""
        if not self.current_words:
            self.segment_start_time = current_time
        self.current_words.append(word)

    def get_current_text(self) -> str:
        """Get the current accumulated text."""
        return " ".join(self.current_words).strip()

    def get_duration(self, current_time: float) -> float:
        """Get duration of current segment."""
        if not self.current_words:
            return 0.0
        return current_time - self.segment_start_time

    def time_since_last_flush(self, current_time: float) -> float:
        """Get time since last segment was displayed."""
        return current_time - self.last_flush_time

    def should_flush(self, current_time: float, vad_pause: bool = False) -> bool:
        """
        Determine if we should flush the current batch.

        Rules:
        1. Must meet minimum duration since last flush (prevent flashing)
        2. Then check: VAD pause OR max duration OR max words exceeded
        """
        if not self.current_words:
            return False

        # Check if enough time passed since last display
        if self.time_since_last_flush(current_time) < self.min_duration:
            return False

        word_count = len(self.current_words)
        duration = self.get_duration(current_time)

        # Check flush conditions
        if vad_pause:
            return True
        if duration >= self.max_duration:
            return True
        if word_count >= self.max_words:
            return True

        return False

    def flush(self, current_time: float) -> SubtitleSegment | None:
        """
        Flush current batch and return the completed segment.
        Note: lines will be set after translation.
        """
        if not self.current_words:
            return None

        text = self.get_current_text()

        segment = SubtitleSegment(
            text=text,
            translated_text="",  # Will be set after translation
            lines=[],  # Will be set after translation
            start_time=self.segment_start_time,
            end_time=current_time,
            word_count=len(self.current_words),
            char_count=len(text),
        )

        self.completed_segments.append(segment)
        self.current_words = []
        self.last_flush_time = current_time

        return segment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time STT with subtitle batching"
    )
    parser.add_argument("--max-steps", default=4096)
    parser.add_argument("--hf-repo")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache downloaded models. If not specified, uses huggingface_hub default (~/.cache/huggingface)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.5,
        help="Minimum duration between subtitle displays (seconds)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=5.0,
        help="Maximum duration for a subtitle segment (seconds)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Maximum number of lines per subtitle (default: 2)",
    )
    parser.add_argument(
        "--words-per-line",
        type=int,
        default=7,
        help="Maximum words per line (default: 7, ~2.3s to read at 180 WPM)",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama API host (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="zongwei/gemma3-translator:4b",
        help="Ollama model to use for translation (default: zongwei/gemma3-translator:4b)",
    )
    parser.add_argument(
        "--target-language",
        type=str,
        default="Spanish",
        help="Target language for translation (default: Spanish)",
    )
    parser.add_argument(
        "--translation-timeout",
        type=float,
        default=3.0,
        help="Timeout for translation API calls in seconds (default: 3.0)",
    )
    args = parser.parse_args()

    # Prepare kwargs for hf_hub_download
    download_kwargs = {}
    if args.cache_dir is not None:
        download_kwargs["cache_dir"] = args.cache_dir
        print(f"Using cache directory: {args.cache_dir}")
        os.makedirs(args.cache_dir, exist_ok=True)
    else:
        # Let huggingface_hub use its default cache
        default_cache = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        print(f"Using default cache directory: {default_cache}")

    if args.hf_repo is None:
        args.hf_repo = "kyutai/stt-1b-en_fr-mlx"

    print(f"Downloading model from {args.hf_repo}")
    lm_config = hf_hub_download(args.hf_repo, "config.json", **download_kwargs)
    with open(lm_config, "r") as fobj:
        lm_config = json.load(fobj)

    print(f"Downloading weights: {lm_config['mimi_name']}")
    mimi_weights = hf_hub_download(args.hf_repo, lm_config["mimi_name"], **download_kwargs)

    moshi_name = lm_config.get("moshi_name", "model.safetensors")
    print(f"Downloading weights: {moshi_name}")
    moshi_weights = hf_hub_download(args.hf_repo, moshi_name, **download_kwargs)

    print(f"Downloading tokenizer: {lm_config['tokenizer_name']}")
    tokenizer = hf_hub_download(args.hf_repo, lm_config["tokenizer_name"], **download_kwargs)

    lm_config = models.LmConfig.from_config_dict(lm_config)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    if moshi_weights.endswith(".q4.safetensors"):
        nn.quantize(model, bits=4, group_size=32)
    elif moshi_weights.endswith(".q8.safetensors"):
        nn.quantize(model, bits=8, group_size=64)

    print(f"loading model weights from {moshi_weights}")
    if args.hf_repo.endswith("-candle"):
        model.load_pytorch_weights(moshi_weights, lm_config, strict=True)
    else:
        model.load_weights(moshi_weights, strict=True)

    print(f"loading the text tokenizer from {tokenizer}")
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer)  # type: ignore

    print(f"loading the audio tokenizer {mimi_weights}")
    generated_codebooks = lm_config.generated_codebooks
    other_codebooks = lm_config.other_codebooks
    mimi_codebooks = max(generated_codebooks, other_codebooks)
    audio_tokenizer = rustymimi.Tokenizer(mimi_weights, num_codebooks=mimi_codebooks)  # type: ignore
    print("warming up the model")
    model.warmup()
    gen = models.LmGen(
        model=model,
        max_steps=args.max_steps,
        text_sampler=utils.Sampler(top_k=25, temp=0),
        audio_sampler=utils.Sampler(top_k=250, temp=0.8),
        check=False,
    )

    # Initialize subtitle batcher
    batcher = SubtitleBatcher(
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_lines=args.max_lines,
        words_per_line=args.words_per_line,
    )

    block_queue = queue.Queue()

    def audio_callback(indata, _frames, _time, _status):
        block_queue.put(indata.copy())

    print("\nSubtitle batching enabled:")
    print(f"  Min duration: {args.min_duration}s")
    print(f"  Max duration: {args.max_duration}s")
    print(f"  Max lines: {args.max_lines}")
    print(f"  Words per line: {args.words_per_line}")
    print(f"  Max words: {batcher.max_words}")
    print("\nTranslation settings:")
    print(f"  Ollama host: {args.ollama_host}")
    print(f"  Model: {args.ollama_model}")
    print(f"  Target language: {args.target_language}")
    print(f"  Timeout: {args.translation_timeout}s")

    # Test translation connection
    print("\nTesting translation...")
    test_text = "Hello world"
    print(f"  Testing with: '{test_text}'")
    test_result = translate_with_ollama(
        test_text,
        target_language=args.target_language,
        ollama_host=args.ollama_host,
        model=args.ollama_model,
        timeout=args.translation_timeout,
    )
    if test_result == test_text:
        print("  ⚠️  Translation failed - will show original text only")
    else:
        print(f"  ✅ Translation works: '{test_result}'")

    print("\nRecording audio from microphone, speak to get subtitles...\n")
    print("=" * 80)

    start_time = time.time()
    word_buffer = ""  # Accumulate token pieces into complete words

    with sd.InputStream(
        channels=1,
        dtype="float32",
        samplerate=24000,
        blocksize=1920,
        callback=audio_callback,
    ):
        try:
            while True:
                block = block_queue.get()
                current_time = time.time() - start_time

                block = block[None, :, 0]
                other_audio_tokens = audio_tokenizer.encode_step(block[None, 0:1])
                other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[
                    :, :, :other_codebooks
                ]

                text_token = gen.step(other_audio_tokens[0])
                text_token = text_token[0].item()

                if text_token not in (0, 3):
                    token_piece = text_tokenizer.id_to_piece(text_token)  # type: ignore

                    # Check if this token starts a new word (▁ is SentencePiece word boundary marker)
                    if token_piece.startswith("▁"):
                        # Flush previous word if exists
                        if word_buffer.strip():
                            batcher.add_word(word_buffer.strip(), current_time)
                        # Start new word (remove ▁ marker)
                        word_buffer = token_piece[1:]  # Remove ▁ character
                    else:
                        # Continue building current word
                        word_buffer += token_piece

                # Check if we should flush the batch (no VAD, just time/char limits)
                if batcher.should_flush(current_time, vad_pause=False):
                    # Flush any remaining word in buffer before flushing segment
                    if word_buffer.strip():
                        batcher.add_word(word_buffer.strip(), current_time)
                        word_buffer = ""

                    segment = batcher.flush(current_time)
                    if segment:
                        # Print original transcription immediately
                        print(f"\nOriginal: {segment.text}")

                        # Translate the segment
                        translated = translate_with_ollama(
                            segment.text,
                            target_language=args.target_language,
                            ollama_host=args.ollama_host,
                            model=args.ollama_model,
                            timeout=args.translation_timeout,
                        )
                        segment.translated_text = translated

                        # Print translation
                        print(f"Translated: {translated}")

                        # Break translated text into lines
                        segment.lines = SubtitleBatcher.break_into_lines(
                            translated, args.max_lines, args.words_per_line
                        )

                        # Display the formatted subtitle (multi-line)
                        print()
                        for line in segment.lines:
                            print(line)
                        print("=" * 80)

        except KeyboardInterrupt:
            print("\n\nStopping...")
            # Flush any remaining word in buffer
            if word_buffer.strip():
                batcher.add_word(word_buffer.strip(), time.time() - start_time)
                word_buffer = ""

            # Flush any remaining words in segment
            if batcher.current_words:
                segment = batcher.flush(time.time() - start_time)
                if segment:
                    # Print original transcription immediately
                    print(f"\nOriginal: {segment.text}")

                    # Translate the segment
                    translated = translate_with_ollama(
                        segment.text,
                        target_language=args.target_language,
                        ollama_host=args.ollama_host,
                        model=args.ollama_model,
                        timeout=args.translation_timeout,
                    )
                    segment.translated_text = translated

                    # Print translation
                    print(f"Translated: {translated}")

                    # Break translated text into lines
                    segment.lines = SubtitleBatcher.break_into_lines(
                        translated, args.max_lines, args.words_per_line
                    )

                    # Display the formatted subtitle
                    print()
                    for line in segment.lines:
                        print(line)
                    print("=" * 80)

            print(f"\n\nTotal segments captured: {len(batcher.completed_segments)}")
            print(f"Segments available for translation pipeline")
