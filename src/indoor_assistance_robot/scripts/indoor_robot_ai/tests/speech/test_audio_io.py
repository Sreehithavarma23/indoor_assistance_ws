"""Unit tests for file I/O. Microphone hardware is not required."""

from pathlib import Path

import numpy as np
import pytest

from src.speech.audio_capture import load_audio_file, save_audio_file
from src.speech.exceptions import AudioLoadError, MicrophoneError
from src.speech.audio_capture import record_microphone
from tests.speech.conftest import silence, write_wav


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(AudioLoadError):
        load_audio_file(tmp_path / "missing.wav")


def test_round_trip_wav(tmp_path: Path):
    path = tmp_path / "tone.wav"
    original = silence(0.25)
    original[10] = 0.5
    write_wav(path, original, 16000)
    loaded, rate = load_audio_file(path)
    assert rate == 16000
    assert loaded.shape == original.shape
    assert np.allclose(loaded, original, atol=1e-4)


def test_save_audio_file_creates_parent(tmp_path: Path):
    path = tmp_path / "nested" / "out.wav"
    save_audio_file(path, silence(0.1), 16000)
    assert path.is_file()


def test_record_microphone_rejects_non_positive_duration():
    with pytest.raises(MicrophoneError):
        record_microphone(duration_seconds=0)
