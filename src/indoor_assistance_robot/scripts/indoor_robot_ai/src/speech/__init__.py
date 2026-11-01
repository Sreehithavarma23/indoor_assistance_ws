"""Speech processing: audio → transcript.

Public interface for later NLU:

    transcribe_file(path) -> Transcript
    transcribe_array(audio, sample_rate) -> Transcript
    Transcript.text

This package does not classify intents or control the robot.
"""

from src.speech.pipeline import transcribe_array, transcribe_file, transcribe_microphone
from src.speech.types import Transcript, TranscriptSegment

__all__ = [
    "Transcript",
    "TranscriptSegment",
    "transcribe_array",
    "transcribe_file",
    "transcribe_microphone",
]
