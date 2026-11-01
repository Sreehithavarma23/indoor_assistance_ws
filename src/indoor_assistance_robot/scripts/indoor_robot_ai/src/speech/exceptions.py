"""Errors raised by the speech-processing module only."""


class SpeechProcessingError(Exception):
    """Base error for the speech pipeline."""


class AudioLoadError(SpeechProcessingError):
    """Audio file is missing, unreadable, or not a supported format."""


class MicrophoneError(SpeechProcessingError):
    """Microphone capture failed or is unavailable."""


class ASRModelError(SpeechProcessingError):
    """Faster-Whisper model could not be loaded or used."""
