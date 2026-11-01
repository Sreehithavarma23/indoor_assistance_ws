"""Load recorded audio and optionally capture from a microphone."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

from src.config_loader import load_speech
from src.logging_utils import get_logger
from src.speech.exceptions import AudioLoadError, MicrophoneError

logger = get_logger("speech.audio_capture")


def load_audio_file(path: str | Path) -> Tuple[np.ndarray, int]:
    """Load an audio file as float32 samples plus its original sample rate.

    Stereo is kept as (samples, channels). Preprocessing converts to mono.
    """
    audio_path = Path(path)
    if not audio_path.is_file():
        raise AudioLoadError(f"Audio file not found: {audio_path}")

    try:
        audio, sample_rate = sf.read(str(audio_path), always_2d=False, dtype="float32")
    except Exception as exc:
        raise AudioLoadError(f"Could not read audio file: {audio_path}") from exc

    if audio.size == 0:
        raise AudioLoadError(f"Audio file is empty: {audio_path}")

    logger.info("Loaded %s (sr=%s, shape=%s)", audio_path.name, sample_rate, audio.shape)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def save_audio_file(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    """Write float audio to disk (WAV/FLAC via soundfile)."""
    audio_path = Path(path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_path), np.asarray(audio, dtype=np.float32), int(sample_rate))
    logger.info("Wrote %s", audio_path)
    return audio_path


def record_microphone(
    duration_seconds: float | None = None,
    sample_rate: int | None = None,
) -> Tuple[np.ndarray, int]:
    """Record mono audio from the default microphone.

    sounddevice/PortAudio must be available. This is optional input, not
    required for file-based tests.
    """
    config = load_speech()
    mic = config.get("microphone", {})
    if duration_seconds is None:
        duration = float(mic.get("default_duration_seconds", 5))
    else:
        duration = float(duration_seconds)
    rate = int(sample_rate or config.get("sample_rate", 16000))
    channels = int(mic.get("channels", 1))

    if duration <= 0:
        raise MicrophoneError("Recording duration must be positive.")

    try:
        import sounddevice as sd
    except Exception as exc:
        raise MicrophoneError(
            "Microphone capture requires sounddevice and the PortAudio library "
            "(Ubuntu: sudo apt install libportaudio2)."
        ) from exc

    logger.info("Recording microphone for %.2fs at %d Hz", duration, rate)
    try:
        recorded = sd.rec(
            int(duration * rate),
            samplerate=rate,
            channels=channels,
            dtype="float32",
        )
        sd.wait()
    except Exception as exc:
        raise MicrophoneError(f"Microphone recording failed: {exc}") from exc

    audio = np.asarray(recorded, dtype=np.float32)
    if audio.ndim == 2 and audio.shape[1] == 1:
        audio = audio[:, 0]
    return audio, rate
