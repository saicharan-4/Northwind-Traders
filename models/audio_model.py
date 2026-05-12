"""
Audio transcription via OpenAI Whisper (local inference).
Converts uploaded audio clips to text, which is then fed into the
text sentiment channel of MoodSyncAI.

Install: pip install openai-whisper
"""

from __future__ import annotations
from typing import Optional


class AudioTranscriber:
    """
    Wraps OpenAI Whisper for offline speech-to-text.

    Supported models (speed vs accuracy):
        "tiny"   – fastest, lower accuracy (good for demo/CPU)
        "base"   – balanced
        "small"  – good accuracy, still runs on CPU
        "medium" – better accuracy, needs GPU
        "large"  – best accuracy, requires GPU with ≥8 GB VRAM
    """

    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Args:
            model_size: Whisper model variant.
            device: 'cpu' | 'cuda' | None (auto-detect).
        """
        import torch, whisper

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = whisper.load_model(model_size, device=self.device)

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to the .wav / .mp3 / .m4a file.
            language:   ISO-639-1 code to force a language (e.g. 'en').
                        None → auto-detect.

        Returns:
            Transcribed text string.
        """
        options = {"fp16": False}          # fp16=False ensures CPU compatibility
        if language:
            options["language"] = language

        result = self.model.transcribe(audio_path, **options)
        return result["text"].strip()

    def transcribe_with_timestamps(self, audio_path: str) -> list[dict]:
        """
        Returns per-segment transcription with start/end timestamps.
        Useful for the emotion timeline feature.

        Returns:
            [{"start": float, "end": float, "text": str}, ...]
        """
        result = self.model.transcribe(audio_path, fp16=False, word_timestamps=True)
        return [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in result["segments"]
        ]
