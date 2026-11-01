"""Validation statuses and the public ValidationResult object."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List

from src.nlu.types import NLUResult


class ValidationStatus(str, Enum):
    """Controlled task-readiness labels. Not robot execution states."""

    VALID = "VALID"
    INCOMPLETE = "INCOMPLETE"
    AMBIGUOUS = "AMBIGUOUS"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class ValidationResult:
    """Whether an NLU interpretation is ready for a later planner.

    This object does not ask clarification questions and does not plan actions.
    """

    original_text: str
    intent: str
    intent_confidence: float
    entities: Dict[str, str]
    status: ValidationStatus
    reasons: List[str] = field(default_factory=list)
    missing_entities: List[str] = field(default_factory=list)
    ambiguous_entities: List[str] = field(default_factory=list)
    intent_probabilities: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_nlu(
        cls,
        nlu: NLUResult,
        status: ValidationStatus,
        reasons: List[str] | None = None,
        missing_entities: List[str] | None = None,
        ambiguous_entities: List[str] | None = None,
    ) -> "ValidationResult":
        return cls(
            original_text=nlu.original_text,
            intent=nlu.intent,
            intent_confidence=nlu.intent_confidence,
            entities=dict(nlu.entities),
            status=status,
            reasons=list(reasons or []),
            missing_entities=list(missing_entities or []),
            ambiguous_entities=list(ambiguous_entities or []),
            intent_probabilities=dict(nlu.intent_probabilities),
        )
