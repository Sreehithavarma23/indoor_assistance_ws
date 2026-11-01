"""Helpers for speech unit tests. Synthetic signals only — not ASR labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> Path:
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sample_rate)
    return path


def silence(seconds: float, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def tone(
    seconds: float,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 0.3,
) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False)
    return (amplitude * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)


@pytest.fixture
def speech_like_clip():
    """Energy-rich clip for VAD tests. This is not human speech."""
    return np.concatenate([silence(0.3), tone(0.6), silence(0.3)])
