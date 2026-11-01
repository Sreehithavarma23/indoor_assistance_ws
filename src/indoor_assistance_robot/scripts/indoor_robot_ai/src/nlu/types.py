"""Structured NLU objects. No validation status and no robot actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class EntityMention:
    """One grounded span found in the text."""

    entity_type: str
    text: str
    value: str
    role: Optional[str] = None
    start: int = -1
    end: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IntentPrediction:
    """Output of an intent classifier implementation."""

    intent: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class NLUExample:
    """One labeled command from the development dataset."""

    text: str
    intent: str
    entities: Dict[str, str] = field(default_factory=dict)
    split: str = ""


@dataclass(frozen=True)
class NLUResult:
    """Public NLU output for later validation / planning stages.

    This phase does not set VALID / INCOMPLETE / AMBIGUOUS.
    intent may be UNSUPPORTED when the text is out of scope or empty.
    """

    original_text: str
    intent: str
    intent_confidence: float
    entities: Dict[str, str] = field(default_factory=dict)
    mentions: List[EntityMention] = field(default_factory=list)
    intent_probabilities: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload
