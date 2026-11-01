"""File/microphone/array audio → Transcript.

This is the public speech-processing entry point. It stops at text.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.config_loader import load_speech
from src.logging_utils import get_logger, setup_logging
from src.speech.asr import WhisperASR
from src.speech.audio_capture import load_audio_file, record_microphone
from src.speech.preprocess import preprocess_audio
from src.speech.types import Transcript
from src.speech.vad import trim_silence

logger = get_logger("speech.pipeline")

_default_asr: WhisperASR | None = None


def _get_asr(asr: Optional[WhisperASR]) -> WhisperASR:
    """Reuse one Faster-Whisper instance so evaluation does not reload the model."""
    global _default_asr
    if asr is not None:
        return asr
    if _default_asr is None:
        _default_asr = WhisperASR()
    return _default_asr


def _empty_transcript(
    *,
    source: str,
    audio_path: Optional[str],
    sample_rate: int,
    pipeline_latency: float,
) -> Transcript:
    return Transcript(
        text="",
        speech_detected=False,
        source=source,
        audio_path=audio_path,
        sample_rate=sample_rate,
        pipeline_latency_seconds=pipeline_latency,
    )


def transcribe_array(
    audio: np.ndarray,
    source_rate: int,
    asr: Optional[WhisperASR] = None,
    source: str = "array",
    audio_path: Optional[str] = None,
) -> Transcript:
    """Preprocess, apply energy VAD, then run Faster-Whisper when speech exists."""
    setup_logging()
    started = time.perf_counter()
    config = load_speech()
    target_rate = int(config.get("sample_rate", 16000))

    prepared = preprocess_audio(
        audio,
        source_rate,
        target_rate,
        settings=config.get("preprocess", {}),
    )

    vad_settings = config.get("vad", {})
    speech_detected = True
    if vad_settings.get("enabled", True):
        prepared, speech_detected = trim_silence(prepared, target_rate, vad_settings)
        if not speech_detected:
            logger.info("No speech detected; skipping ASR.")
            return _empty_transcript(
                source=source,
                audio_path=audio_path,
                sample_rate=target_rate,
                pipeline_latency=time.perf_counter() - started,
            )

    transcript = _get_asr(asr).transcribe(
        prepared,
        target_rate,
        source=source,
        audio_path=audio_path,
    )
    return Transcript(
        text=transcript.text,
        language=transcript.language,
        segments=transcript.segments,
        latency_seconds=transcript.latency_seconds,
        pipeline_latency_seconds=time.perf_counter() - started,
        source=source,
        audio_path=audio_path,
        speech_detected=speech_detected,
        sample_rate=target_rate,
    )


def transcribe_file(
    path: str | Path,
    asr: Optional[WhisperASR] = None,
) -> Transcript:
    """Transcribe a recorded audio file. This is the primary testable path."""
    audio, source_rate = load_audio_file(path)
    return transcribe_array(
        audio,
        source_rate,
        asr=asr,
        source="file",
        audio_path=str(Path(path)),
    )


def transcribe_microphone(
    duration_seconds: float | None = None,
    asr: Optional[WhisperASR] = None,
) -> Transcript:
    """Record from the default microphone, then run the same pipeline."""
    audio, source_rate = record_microphone(duration_seconds=duration_seconds)
    return transcribe_array(
        audio,
        source_rate,
        asr=asr,
        source="microphone",
    )
