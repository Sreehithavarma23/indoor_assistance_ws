"""Faster-Whisper ASR wrapper.

This module converts a waveform into text. It must not classify intents
or emit robot actions.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional

import numpy as np

from src.config_loader import PROJECT_ROOT, load_speech
from src.logging_utils import get_logger
from src.speech.exceptions import ASRModelError
from src.speech.types import Transcript, TranscriptSegment

logger = get_logger("speech.asr")


def resolve_device(device: str) -> str:
    """Map 'auto' to cuda when a GPU is visible, otherwise cpu."""
    requested = (device or "auto").lower()
    if requested != "auto":
        return requested
    try:
        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def resolve_compute_type(compute_type: str, device: str) -> str:
    """Choose a Faster-Whisper compute type if the config says 'default'."""
    requested = (compute_type or "default").lower()
    if requested != "default":
        return requested
    return "float16" if device == "cuda" else "int8"


class WhisperASR:
    """Lazy Faster-Whisper model. Safe to construct in tests without loading."""

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        config = load_speech()
        self.settings = dict(config.get("asr", {}))
        if settings:
            self.settings.update(dict(settings))
        self._model: Any = None
        self.model_load_seconds: Optional[float] = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        model_size = str(self.settings.get("model_size", "base"))
        device = resolve_device(str(self.settings.get("device", "auto")))
        compute_type = resolve_compute_type(
            str(self.settings.get("compute_type", "default")),
            device,
        )
        download_root = PROJECT_ROOT / str(
            self.settings.get("download_root", "models/whisper")
        )
        download_root.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Loading Faster-Whisper model=%s device=%s compute_type=%s",
            model_size,
            device,
            compute_type,
        )
        started = time.perf_counter()
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ASRModelError(
                "faster-whisper is not installed. Run: pip install -r requirements.txt"
            ) from exc

        try:
            self._model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=str(download_root),
            )
        except Exception as exc:
            raise ASRModelError(f"Failed to load Faster-Whisper model: {exc}") from exc
        self.model_load_seconds = time.perf_counter() - started
        return self._model

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        source: str = "array",
        audio_path: Optional[str] = None,
    ) -> Transcript:
        """Transcribe a mono float32 waveform already at the model sample rate."""
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return Transcript(
                text="",
                speech_detected=False,
                source=source,
                audio_path=audio_path,
                sample_rate=sample_rate,
            )

        model = self._load_model()
        language = self.settings.get("language") or None
        started = time.perf_counter()
        try:
            segments_iter, info = model.transcribe(
                samples,
                language=language,
                beam_size=int(self.settings.get("beam_size", 5)),
                vad_filter=bool(self.settings.get("vad_filter", True)),
            )
            collected = list(segments_iter)
        except Exception as exc:
            raise ASRModelError(f"Faster-Whisper transcription failed: {exc}") from exc
        latency = time.perf_counter() - started

        segments = [
            TranscriptSegment(
                start=float(segment.start),
                end=float(segment.end),
                text=segment.text.strip(),
            )
            for segment in collected
        ]
        text = " ".join(part.text for part in segments).strip()
        detected_language = getattr(info, "language", language)

        logger.info(
            "ASR transcript=%r language=%s latency=%.3fs",
            text,
            detected_language,
            latency,
        )
        return Transcript(
            text=text,
            language=detected_language,
            segments=segments,
            latency_seconds=latency,
            source=source,
            audio_path=audio_path,
            speech_detected=True,
            sample_rate=sample_rate,
        )
