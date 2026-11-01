"""ASR tests.

Unit tests never download a model.
Real Faster-Whisper tests run only when RUN_ASR_TESTS=1.
"""

import os
from pathlib import Path

import pytest

from src.speech.asr import WhisperASR, resolve_compute_type, resolve_device
from src.speech.pipeline import transcribe_file
from tests.speech.conftest import write_wav, tone, silence
import numpy as np


def test_resolve_device_respects_explicit_value():
    assert resolve_device("cpu") == "cpu"


def test_resolve_compute_type_defaults():
    assert resolve_compute_type("default", "cpu") == "int8"
    assert resolve_compute_type("default", "cuda") == "float16"
    assert resolve_compute_type("int8", "cuda") == "int8"


def test_whisper_asr_can_be_constructed_without_loading_model():
    engine = WhisperASR()
    assert engine._model is None


@pytest.mark.asr
@pytest.mark.skipif(
    os.environ.get("RUN_ASR_TESTS") != "1",
    reason="Real ASR tests are opt-in: RUN_ASR_TESTS=1",
)
def test_real_asr_requires_provided_speech_file(tmp_path: Path):
    """Transcribe a user-provided recording. Does not fabricate a reference.

    Set ASR_TEST_AUDIO to a real speech wav. The test only checks that the
    pipeline returns a Transcript object. It does not assert a specific sentence.
    """
    audio_path = os.environ.get("ASR_TEST_AUDIO")
    if not audio_path:
        pytest.skip("Set ASR_TEST_AUDIO to a real speech recording.")
    result = transcribe_file(audio_path)
    assert result.source == "file"
    assert result.latency_seconds >= 0.0


@pytest.mark.asr
@pytest.mark.skipif(
    os.environ.get("RUN_ASR_TESTS") != "1",
    reason="Real ASR tests are opt-in: RUN_ASR_TESTS=1",
)
def test_real_asr_on_tone_does_not_claim_accuracy(tmp_path: Path):
    """Smoke-load the model on a tone. Output is not treated as a correct transcript."""
    path = write_wav(
        tmp_path / "tone.wav",
        np.concatenate([silence(0.2), tone(1.0), silence(0.2)]),
    )
    engine = WhisperASR({"model_size": "tiny", "device": "cpu", "vad_filter": False})
    result = transcribe_file(path, asr=engine)
    assert hasattr(result, "text")
    assert result.pipeline_latency_seconds >= 0.0
