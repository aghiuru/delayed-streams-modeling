"""
Real-time transcription engine with subtitle batching.

This module handles the core STT functionality, including model loading,
audio processing, and subtitle generation.
"""

import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Callable, Optional

import mlx.core as mx
import mlx.nn as nn
import requests
import rustymimi
import sentencepiece
import sounddevice as sd
from huggingface_hub import hf_hub_download
from moshi_mlx import models, utils

logger = logging.getLogger(__name__)


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

    prompt = f"""Translate to {target_language}. Output only the translation, no explanations.

{text}"""

    logger.debug(f"Translating: '{text}' to {target_language}")
    start_time = time.time()

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
            elapsed = time.time() - start_time
            if translated:
                logger.debug(f"Translated: '{translated}' ({elapsed:.2f}s)")
                return translated
            else:
                logger.warning(f"Empty translation response ({elapsed:.2f}s)")
                return text
        else:
            logger.warning(f"Translation API returned status {response.status_code}")
            return text

    except requests.exceptions.Timeout:
        logger.warning(f"Translation timeout after {timeout}s")
        return text
    except requests.exceptions.ConnectionError:
        logger.warning("Cannot connect to Ollama")
        return text
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text


class SubtitleBatcher:
    """Batches words into subtitle segments with real-time constraints."""

    def __init__(
        self,
        min_duration: float = 1.5,
        max_duration: float = 5.0,
        max_lines: int = 2,
        words_per_line: int = 7,
        soft_max_words: int = 10,
    ):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_lines = max_lines
        self.words_per_line = words_per_line
        self.soft_max_words = soft_max_words
        self.max_words = max_lines * words_per_line

        self.current_words: List[str] = []
        self.segment_start_time: float = 0.0
        self.last_flush_time: float = 0.0
        self.completed_segments: List[SubtitleSegment] = []

    @staticmethod
    def break_into_lines(text: str, max_lines: int, words_per_line: int) -> List[str]:
        """Break text into multiple lines based on word count."""
        if not text:
            return []

        words = text.strip().split()
        if not words:
            return []

        if len(words) <= words_per_line:
            return [text]

        lines = []
        current_line_words = []

        for i, word in enumerate(words):
            current_line_words.append(word)

            if len(current_line_words) >= words_per_line:
                lines.append(" ".join(current_line_words))
                current_line_words = []

                if len(lines) >= max_lines:
                    remaining_words = words[i + 1 :]
                    if remaining_words:
                        lines[-1] = lines[-1] + " " + " ".join(remaining_words)
                    return lines

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

    def is_sentence_boundary(self) -> bool:
        """Check if the last word ends with sentence-ending punctuation."""
        if not self.current_words:
            return False
        last_word = self.current_words[-1].rstrip()
        return last_word.endswith((".", "!", "?"))

    def should_flush(self, current_time: float, vad_pause: bool = False) -> bool:
        """Determine if we should flush the current batch."""
        if not self.current_words:
            return False

        if self.time_since_last_flush(current_time) < self.min_duration:
            return False

        word_count = len(self.current_words)
        duration = self.get_duration(current_time)

        if vad_pause:
            return True
        if duration >= self.max_duration:
            return True
        if word_count >= self.max_words:
            return True

        if word_count >= self.soft_max_words:
            if self.is_sentence_boundary():
                return True

        return False

    def flush(self, current_time: float) -> SubtitleSegment | None:
        """Flush current batch and return the completed segment."""
        if not self.current_words:
            return None

        text = self.get_current_text()

        segment = SubtitleSegment(
            text=text,
            translated_text="",
            lines=[],
            start_time=self.segment_start_time,
            end_time=current_time,
            word_count=len(self.current_words),
            char_count=len(text),
        )

        self.completed_segments.append(segment)
        self.current_words = []
        self.last_flush_time = current_time

        return segment


class TranscriptionEngine:
    """Manages the transcription process in a separate thread."""

    def __init__(self, config):
        """Initialize the transcription engine with configuration."""
        self.config = config
        self.model = None
        self.lm_config = None
        self.audio_tokenizer = None
        self.mimi_weights_path = None  # Store for reinitializing tokenizer
        self.text_tokenizer = None
        self.gen = None
        self.batcher = None
        self.is_running = False
        self.thread = None
        self.block_queue = queue.Queue()
        self.stream = None

        # Callback for when a new subtitle segment is ready
        self.on_subtitle_callback: Optional[Callable[[SubtitleSegment], None]] = None

    def load_models(self) -> bool:
        """Load the STT models. Returns True if successful."""
        try:
            logger.info(f"Loading models from {self.config.hf_repo}")

            # Prepare download kwargs
            download_kwargs = {}
            if self.config.cache_dir:
                download_kwargs["cache_dir"] = self.config.cache_dir
                os.makedirs(self.config.cache_dir, exist_ok=True)
                logger.debug(f"Using cache directory: {self.config.cache_dir}")

            # Download config
            logger.debug("Downloading config.json")
            lm_config_path = hf_hub_download(
                self.config.hf_repo, "config.json", **download_kwargs
            )
            with open(lm_config_path, "r") as f:
                lm_config_dict = json.load(f)

            # Download weights
            logger.debug(f"Downloading weights: {lm_config_dict['mimi_name']}")
            mimi_weights = hf_hub_download(
                self.config.hf_repo, lm_config_dict["mimi_name"], **download_kwargs
            )
            moshi_name = lm_config_dict.get("moshi_name", "model.safetensors")
            logger.debug(f"Downloading weights: {moshi_name}")
            moshi_weights = hf_hub_download(
                self.config.hf_repo, moshi_name, **download_kwargs
            )
            logger.debug(f"Downloading tokenizer: {lm_config_dict['tokenizer_name']}")
            tokenizer_path = hf_hub_download(
                self.config.hf_repo,
                lm_config_dict["tokenizer_name"],
                **download_kwargs,
            )

            # Create model
            logger.debug("Creating model")
            self.lm_config = models.LmConfig.from_config_dict(lm_config_dict)
            self.model = models.Lm(self.lm_config)
            self.model.set_dtype(mx.bfloat16)

            # Quantize if needed
            if moshi_weights.endswith(".q4.safetensors"):
                logger.debug("Quantizing model to 4-bit")
                nn.quantize(self.model, bits=4, group_size=32)
            elif moshi_weights.endswith(".q8.safetensors"):
                logger.debug("Quantizing model to 8-bit")
                nn.quantize(self.model, bits=8, group_size=64)

            # Load weights
            logger.debug(f"Loading model weights from {moshi_weights}")
            if self.config.hf_repo.endswith("-candle"):
                self.model.load_pytorch_weights(moshi_weights, self.lm_config, strict=True)
            else:
                self.model.load_weights(moshi_weights, strict=True)

            # Load tokenizers
            logger.debug(f"Loading text tokenizer from {tokenizer_path}")
            self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

            logger.debug(f"Loading audio tokenizer from {mimi_weights}")
            self.mimi_weights_path = mimi_weights  # Store for later reinitialization
            generated_codebooks = self.lm_config.generated_codebooks
            other_codebooks = self.lm_config.other_codebooks
            mimi_codebooks = max(generated_codebooks, other_codebooks)
            self.audio_tokenizer = rustymimi.Tokenizer(
                mimi_weights, num_codebooks=mimi_codebooks
            )

            # Warmup
            logger.debug("Warming up model")
            self.model.warmup()

            # Create generator
            logger.debug("Creating generator")
            self.gen = models.LmGen(
                model=self.model,
                max_steps=self.config.max_steps,
                text_sampler=utils.Sampler(top_k=25, temp=0),
                audio_sampler=utils.Sampler(top_k=250, temp=0.8),
                check=False,
            )

            # Create batcher
            logger.debug("Creating subtitle batcher")
            self.batcher = SubtitleBatcher(
                min_duration=self.config.min_duration,
                max_duration=self.config.max_duration,
                max_lines=self.config.max_lines,
                words_per_line=self.config.words_per_line,
                soft_max_words=self.config.soft_max_words,
            )

            logger.info("Models loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"Error loading models: {e}", exc_info=True)
            return False

    def audio_callback(self, indata, _frames, _time, _status):
        """Callback for audio input stream."""
        if self.is_running:
            self.block_queue.put(indata.copy())

    def start_recording(self):
        """Start the transcription process."""
        if self.is_running:
            logger.warning("Recording already started")
            return

        if not self.model:
            if not self.load_models():
                logger.error("Failed to load models, cannot start recording")
                return

        logger.info("Starting recording...")
        self.is_running = True
        self.thread = threading.Thread(target=self._transcription_loop, daemon=True)
        self.thread.start()

        # Start audio stream
        device = self.config.audio_device if self.config.audio_device else None
        logger.debug(f"Starting audio stream (device: {device})")
        self.stream = sd.InputStream(
            device=device,
            channels=1,
            dtype="float32",
            samplerate=24000,
            blocksize=1920,
            callback=self.audio_callback,
        )
        self.stream.start()
        logger.info("Recording started successfully")

    def stop_recording(self):
        """Stop the transcription process."""
        logger.info("Stopping recording...")
        self.is_running = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            logger.debug("Audio stream stopped")

        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
            logger.debug("Transcription thread stopped")

        logger.info("Recording stopped")

    def _transcription_loop(self):
        """Main transcription loop running in a separate thread."""
        logger.info("Transcription loop started")
        start_time = time.time()
        word_buffer = ""
        block_count = 0

        other_codebooks = self.lm_config.other_codebooks

        while self.is_running:
            try:
                block = self.block_queue.get(timeout=0.1)
                block_count += 1
            except queue.Empty:
                continue

            try:
                current_time = time.time() - start_time

                # Process audio
                block = block[None, :, 0]
                other_audio_tokens = self.audio_tokenizer.encode_step(block[None, 0:1])
                other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[
                    :, :, :other_codebooks
                ]

                # Generate text token
                text_token = self.gen.step(other_audio_tokens[0])
                text_token = text_token[0].item()

            except ValueError as e:
                if "narrow invalid args" in str(e):
                    logger.warning(f"Audio tokenizer buffer overflow at block {block_count}, reinitializing...")

                    # Flush any pending words/segments before resetting
                    if word_buffer.strip():
                        logger.info(f"Flushing pending word buffer: '{word_buffer.strip()}'")
                        self.batcher.add_word(word_buffer.strip(), current_time)
                        word_buffer = ""

                    if self.batcher.current_words:
                        logger.info(f"Flushing pending segment with {len(self.batcher.current_words)} words")
                        segment = self.batcher.flush(current_time)
                        if segment and self.on_subtitle_callback:
                            # Translate and send the pending segment
                            translated = translate_with_ollama(
                                segment.text,
                                target_language=self.config.target_language,
                                ollama_host=self.config.ollama_host,
                                model=self.config.ollama_model,
                                timeout=self.config.translation_timeout,
                            )
                            segment.translated_text = translated
                            segment.lines = SubtitleBatcher.break_into_lines(
                                translated, self.config.max_lines, self.config.words_per_line
                            )
                            self.on_subtitle_callback(segment)

                    # Reinitialize the audio tokenizer to reset its internal state
                    try:
                        generated_codebooks = self.lm_config.generated_codebooks
                        other_codebooks_count = self.lm_config.other_codebooks
                        mimi_codebooks = max(generated_codebooks, other_codebooks_count)

                        # Recreate the audio tokenizer
                        logger.info("Recreating audio tokenizer...")
                        self.audio_tokenizer = rustymimi.Tokenizer(
                            self.mimi_weights_path, num_codebooks=mimi_codebooks
                        )
                        logger.info("Audio tokenizer successfully reinitialized")

                        # Reset the generator state as well
                        logger.info("Resetting generator state...")
                        self.gen = models.LmGen(
                            model=self.model,
                            max_steps=self.config.max_steps,
                            text_sampler=utils.Sampler(top_k=25, temp=0),
                            audio_sampler=utils.Sampler(top_k=250, temp=0.8),
                            check=False,
                        )
                        logger.info("Generator successfully reset")

                        # Reset word buffer
                        word_buffer = ""
                        logger.info("Transcription resumed after reinitialization")

                        # Continue with next block
                        continue
                    except Exception as reinit_error:
                        logger.error(f"Failed to recover from tokenizer error: {reinit_error}", exc_info=True)
                        continue
                else:
                    logger.error(f"Unexpected error in transcription loop: {e}", exc_info=True)
                    continue

            # Process token
            if text_token not in (0, 3):
                token_piece = self.text_tokenizer.id_to_piece(text_token)
                logger.debug(f"Token {text_token}: '{token_piece}'")

                if token_piece.startswith("▁"):
                    if word_buffer.strip():
                        logger.debug(f"Adding word: '{word_buffer.strip()}' at {current_time:.2f}s")
                        self.batcher.add_word(word_buffer.strip(), current_time)
                    word_buffer = token_piece[1:]
                else:
                    word_buffer += token_piece

            # Check if should flush
            if self.batcher.should_flush(current_time, vad_pause=False):
                if word_buffer.strip():
                    logger.debug(f"Adding final word: '{word_buffer.strip()}'")
                    self.batcher.add_word(word_buffer.strip(), current_time)
                    word_buffer = ""

                segment = self.batcher.flush(current_time)
                if segment:
                    logger.info(f"Segment flushed: '{segment.text}' ({segment.start_time:.2f}s - {segment.end_time:.2f}s, {segment.word_count} words)")

                    if self.on_subtitle_callback:
                        # Translate
                        translated = translate_with_ollama(
                            segment.text,
                            target_language=self.config.target_language,
                            ollama_host=self.config.ollama_host,
                            model=self.config.ollama_model,
                            timeout=self.config.translation_timeout,
                        )
                        segment.translated_text = translated

                        # Break into lines
                        segment.lines = SubtitleBatcher.break_into_lines(
                            translated, self.config.max_lines, self.config.words_per_line
                        )
                        logger.debug(f"Subtitle lines: {segment.lines}")

                        # Notify callback
                        logger.debug("Invoking subtitle callback")
                        self.on_subtitle_callback(segment)
                    else:
                        logger.warning("No subtitle callback registered!")

        logger.info(f"Transcription loop ended (processed {block_count} audio blocks)")
