"""Unit tests for energy VAD. Uses tones and silence, not a speech model."""

import numpy as np

from src.speech.vad import detect_speech_segments, has_speech, trim_silence
from tests.speech.conftest import silence, tone


def test_silence_has_no_speech():
    audio = silence(1.0)
    assert has_speech(audio, 16000) is False
    assert detect_speech_segments(audio, 16000) == []


def test_continuous_tone_is_not_dropped_by_adaptive_threshold():
    audio = tone(0.8)
    assert has_speech(audio, 16000) is True


def test_loud_tone_is_detected_as_speech_like_energy():
    audio = np.concatenate([silence(0.4), tone(0.5), silence(0.4)])
    assert has_speech(audio, 16000) is True
    segments = detect_speech_segments(audio, 16000)
    assert len(segments) >= 1
    start, end = segments[0]
    assert end > start


def test_trim_silence_keeps_energy_region():
    audio = np.concatenate([silence(0.4), tone(0.5), silence(0.4)])
    trimmed, detected = trim_silence(audio, 16000)
    assert detected is True
    assert len(trimmed) < len(audio)
    assert len(trimmed) >= int(0.5 * 16000)


def test_trim_silence_on_quiet_audio_reports_no_speech():
    trimmed, detected = trim_silence(silence(0.5), 16000)
    assert detected is False
    assert len(trimmed) == int(0.5 * 16000)
