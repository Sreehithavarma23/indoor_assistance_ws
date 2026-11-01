"""Justified audio cleanup for ASR: mono, resample, DC removal, peak normalize.

No learned denoiser is used. Linear interpolation is the resampler so this
module stays dependency-light and easy to test.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.logging_utils import get_logger

logger = get_logger("speech.preprocess")

_EPS = 1e-8


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Average channels to a 1-D float32 waveform."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 1:
        return samples
    if samples.ndim == 2:
        # soundfile uses (frames, channels)
        if samples.shape[1] <= 8:
            return samples.mean(axis=1).astype(np.float32)
        return samples.mean(axis=0).astype(np.float32)
    raise ValueError(f"Unsupported audio shape: {samples.shape}")


def remove_dc(audio: np.ndarray) -> np.ndarray:
    """Subtract the mean so a constant bias does not waste headroom."""
    samples = np.asarray(audio, dtype=np.float32)
    return samples - np.mean(samples)


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample with linear interpolation.

    Chosen as the simplest defensible method. Command-length clips are short,
    so a high-quality resampler is not required yet.
    """
    samples = np.asarray(audio, dtype=np.float32)
    if source_rate == target_rate:
        return samples
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive.")
    if samples.size == 0:
        return samples

    duration = samples.shape[0] / float(source_rate)
    target_length = max(1, int(round(duration * target_rate)))
    source_x = np.linspace(0.0, 1.0, num=samples.shape[0], endpoint=False)
    target_x = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(target_x, source_x, samples).astype(np.float32)


def peak_normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Scale so the max absolute sample is target_peak. Silence is left unchanged."""
    samples = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak < _EPS:
        return samples
    return (samples * (float(target_peak) / peak)).astype(np.float32)


def preprocess_audio(
    audio: np.ndarray,
    source_rate: int,
    target_rate: int,
    settings: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Apply the configured preprocess chain and return 16-bit-range float audio."""
    options = dict(settings or {})
    samples = to_mono(audio)
    samples = resample(samples, source_rate, target_rate)
    if options.get("remove_dc", True):
        samples = remove_dc(samples)
    if options.get("normalize", True):
        samples = peak_normalize(samples, float(options.get("target_peak", 0.95)))
    samples = np.clip(samples, -1.0, 1.0).astype(np.float32)
    logger.debug("Preprocessed audio: %d samples at %d Hz", samples.size, target_rate)
    return samples
