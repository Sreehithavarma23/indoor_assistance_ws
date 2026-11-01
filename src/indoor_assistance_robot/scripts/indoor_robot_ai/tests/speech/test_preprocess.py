"""Unit tests for audio preprocessing. No ASR model is loaded."""

import numpy as np

from src.speech.preprocess import (
    peak_normalize,
    preprocess_audio,
    remove_dc,
    resample,
    to_mono,
)


def test_to_mono_averages_stereo_channels():
    stereo = np.array([[1.0, -1.0], [1.0, -1.0]], dtype=np.float32)
    mono = to_mono(stereo)
    assert mono.shape == (2,)
    assert np.allclose(mono, 0.0)


def test_remove_dc_centers_signal():
    shifted = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    centered = remove_dc(shifted)
    assert abs(float(np.mean(centered))) < 1e-6


def test_resample_changes_length_proportionally():
    audio = np.ones(16000, dtype=np.float32)
    resampled = resample(audio, 16000, 8000)
    assert len(resampled) == 8000


def test_resample_same_rate_is_identity():
    audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    assert np.array_equal(resample(audio, 16000, 16000), audio)


def test_peak_normalize_scales_and_leaves_silence():
    audio = np.array([0.1, -0.2], dtype=np.float32)
    normalized = peak_normalize(audio, target_peak=1.0)
    assert np.isclose(np.max(np.abs(normalized)), 1.0)
    zeros = np.zeros(4, dtype=np.float32)
    assert np.array_equal(peak_normalize(zeros), zeros)


def test_preprocess_audio_returns_mono_float32_at_target_rate():
    stereo = np.stack([np.ones(8000), np.ones(8000)], axis=1).astype(np.float32)
    out = preprocess_audio(stereo, source_rate=8000, target_rate=16000)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert out.size == 16000
    assert float(np.max(np.abs(out))) <= 1.0
