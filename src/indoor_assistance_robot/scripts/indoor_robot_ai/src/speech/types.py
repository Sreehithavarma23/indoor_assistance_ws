"""Data objects produced by speech processing.

NLU should consume Transcript.text. This module must not add intents,
entities, or robot actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class TranscriptSegment:
    """One timed span from the ASR model."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    """Public speech-pipeline output for a later NLU stage."""

    text: str
    language: Optional[str] = None
    segments: List[TranscriptSegment] = field(default_factory=list)
    latency_seconds: float = 0.0
    pipeline_latency_seconds: float = 0.0
    source: str = "file"
    audio_path: Optional[str] = None
    speech_detected: bool = True
    sample_rate: int = 16000
