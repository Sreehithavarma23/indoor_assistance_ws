"""Dialogue statuses, state, and result objects for Phase 5."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.nlu.types import NLUResult
from src.validation.types import ValidationResult


class DialogueStatus(str, Enum):
    """Outcome of one dialogue turn. Not a robot execution state."""

    READY = "READY"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    UNSUPPORTED = "UNSUPPORTED"
    CLARIFICATION_FAILED = "CLARIFICATION_FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class DialogueTurn:
    """One bounded history entry."""

    role: str  # "user" | "system"
    text: str


@dataclass
class DialogueState:
    """Mutable multi-turn state held by DialogueManager.

    Stores structured task data only — not string-concatenated utterances.
    """

    awaiting_clarification: bool = False
    active_nlu: Optional[NLUResult] = None
    last_validation: Optional[ValidationResult] = None
    pending_slots: List[str] = field(default_factory=list)
    last_question: Optional[str] = None
    clarification_count: int = 0
    turns: List[DialogueTurn] = field(default_factory=list)

    def reset(self) -> None:
        self.awaiting_clarification = False
        self.active_nlu = None
        self.last_validation = None
        self.pending_slots = []
        self.last_question = None
        self.clarification_count = 0
        self.turns = []


@dataclass(frozen=True)
class DialogueResult:
    """Public Phase 5 output for a future planner or interactive CLI."""

    status: DialogueStatus
    question: Optional[str]
    pending_slots: List[str]
    task: Dict[str, Any]
    validation: Optional[ValidationResult]
    clarification_count: int
    original_text: str
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "status": self.status.value,
            "question": self.question,
            "pending_slots": list(self.pending_slots),
            "task": dict(self.task),
            "clarification_count": self.clarification_count,
            "original_text": self.original_text,
            "message": self.message,
            "validation": self.validation.to_dict() if self.validation else None,
        }
        return payload


def task_from_nlu(nlu: Optional[NLUResult]) -> Dict[str, Any]:
    """Compact task view for a future planner."""
    if nlu is None:
        return {}
    return {
        "intent": nlu.intent,
        "entities": dict(nlu.entities),
        "intent_confidence": nlu.intent_confidence,
        "original_text": nlu.original_text,
    }
