"""Energy-based voice activity detection.

This is a baseline, not a neural VAD. It is used to:
- reject recordings that are only silence
- trim leading and trailing silence before ASR

It does not interpret language. Faster-Whisper can still apply its own
Silero VAD via asr.vad_filter.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

import numpy as np

from src.logging_utils import get_logger

logger = get_logger("speech.vad")

SpeechSegment = Tuple[int, int]


def _frame_rms(audio: np.ndarray, frame_size: int) -> np.ndarray:
    usable = audio[: len(audio) - (len(audio) % frame_size)]
    if usable.size == 0:
        return np.array([], dtype=np.float32)
    frames = usable.reshape(-1, frame_size)
    return np.sqrt(np.mean(np.square(frames), axis=1)).astype(np.float32)


def detect_speech_segments(
    audio: np.ndarray,
    sample_rate: int,
    settings: Mapping[str, Any] | None = None,
) -> List[SpeechSegment]:
    """Return (start, end) sample indices that look like speech."""
    options = dict(settings or {})
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame_ms = float(options.get("frame_ms", 30))
    frame_size = max(1, int(sample_rate * frame_ms / 1000.0))
    energies = _frame_rms(samples, frame_size)
    if energies.size == 0:
        return []

    floor = float(options.get("energy_floor", 0.01))
    multiplier = float(options.get("energy_multiplier", 3.0))
    max_threshold = float(options.get("energy_max_threshold", 0.08))
    adaptive = float(np.percentile(energies, 20)) * multiplier
    threshold = max(floor, min(adaptive, max_threshold))
    speech_flags = energies >= threshold

    raw: List[SpeechSegment] = []
    start = None
    for index, is_speech in enumerate(speech_flags):
        if is_speech and start is None:
            start = index * frame_size
        elif not is_speech and start is not None:
            raw.append((start, index * frame_size))
            start = None
    if start is not None:
        raw.append((start, min(len(samples), len(speech_flags) * frame_size)))

    pad = int(sample_rate * float(options.get("pad_ms", 200)) / 1000.0)
    min_speech = int(sample_rate * float(options.get("min_speech_ms", 150)) / 1000.0)
    merge_gap = int(sample_rate * float(options.get("merge_gap_ms", 300)) / 1000.0)

    padded: List[SpeechSegment] = []
    for seg_start, seg_end in raw:
        if seg_end - seg_start < min_speech:
            continue
        padded.append((max(0, seg_start - pad), min(len(samples), seg_end + pad)))

    merged: List[SpeechSegment] = []
    for seg in padded:
        if not merged or seg[0] - merged[-1][1] > merge_gap:
            merged.append(seg)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))

    logger.debug("VAD found %d speech segment(s)", len(merged))
    return merged


def has_speech(
    audio: np.ndarray,
    sample_rate: int,
    settings: Mapping[str, Any] | None = None,
) -> bool:
    """True if at least one speech-like segment is present."""
    return bool(detect_speech_segments(audio, sample_rate, settings))


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    settings: Mapping[str, Any] | None = None,
) -> Tuple[np.ndarray, bool]:
    """Keep audio from the first to last detected speech segment.

    Returns (trimmed_audio, speech_detected). If nothing is found, the
    original array is returned unchanged and speech_detected is False.
    Mid-utterance pauses are kept so we do not cut a command in half.
    """
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    segments = detect_speech_segments(samples, sample_rate, settings)
    if not segments:
        return samples, False
    start = segments[0][0]
    end = segments[-1][1]
    return samples[start:end], True


def concatenate_segments(
    audio: np.ndarray,
    segments: Sequence[SpeechSegment],
) -> np.ndarray:
    """Join selected segments. Used by tests; the live pipeline trims instead."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not segments:
        return samples
    return np.concatenate([samples[start:end] for start, end in segments])
