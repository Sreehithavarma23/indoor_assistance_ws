"""Pipeline tests that do not download or run Faster-Whisper."""

from pathlib import Path

import pytest

from src.speech.pipeline import transcribe_array, transcribe_file
from src.speech.types import Transcript
from tests.speech.conftest import silence, write_wav


class ExplodingASR:
    """Used only to prove silence never reaches ASR. Not a fake transcript."""

    def transcribe(self, *args, **kwargs):
        raise AssertionError("ASR must not run when VAD finds no speech")


def test_silence_file_returns_empty_transcript_without_asr(tmp_path: Path):
    path = write_wav(tmp_path / "silence.wav", silence(0.8), 16000)
    result = transcribe_file(path, asr=ExplodingASR())
    assert isinstance(result, Transcript)
    assert result.text == ""
    assert result.speech_detected is False
    assert result.source == "file"
    assert result.audio_path == str(path)


def test_missing_file_raises(tmp_path: Path):
    from src.speech.exceptions import AudioLoadError

    with pytest.raises(AudioLoadError):
        transcribe_file(tmp_path / "nope.wav", asr=ExplodingASR())


def test_silence_array_skips_asr():
    result = transcribe_array(silence(0.6), 16000, asr=ExplodingASR(), source="array")
    assert result.text == ""
    assert result.speech_detected is False
    assert result.pipeline_latency_seconds >= 0.0
