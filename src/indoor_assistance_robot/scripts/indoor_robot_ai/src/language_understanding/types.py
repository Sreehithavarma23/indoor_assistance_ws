"""Typed language-understanding result between dialogue and planning.

This module does not call an LLM or plan robot actions. It is a structured
view of the existing NLU, validation, and dialogue outputs so future language
models can be evaluated behind the same contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EntityProvenance(str, Enum):
    """Where an interpreted entity came from."""

    EXPLICIT_UTTERANCE = "explicit_utterance"
    DIALOGUE_HISTORY = "dialogue_history"
    PERCEPTION = "perception"
    DEFAULT = "default"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EntityInterpretation:
    """A canonical entity with a traceable source."""

    slot: str
    value: str
    provenance: EntityProvenance = EntityProvenance.UNKNOWN
    evidence: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["provenance"] = self.provenance.value
        return payload


@dataclass(frozen=True)
class LanguageUnderstandingResult:
    """Validated task interpretation for a single dialogue turn."""

    original_utterance: str
    interpreted_request: str
    task_category: Optional[str]
    entities: Dict[str, str] = field(default_factory=dict)
    entity_provenance: Dict[str, EntityInterpretation] = field(default_factory=dict)
    missing_information: List[str] = field(default_factory=list)
    ambiguity: List[str] = field(default_factory=list)
    clarification_required: bool = False
    clarification_question: Optional[str] = None
    relevant_references: List[str] = field(default_factory=list)
    proposed_symbolic_task: Optional[Dict[str, Any]] = None
    unsupported_capabilities: List[str] = field(default_factory=list)
    validation_status: Optional[str] = None
    dialogue_status: Optional[str] = None
    model_backend: str = "tfidf_logreg_baseline"
    model_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_utterance": self.original_utterance,
            "interpreted_request": self.interpreted_request,
            "task_category": self.task_category,
            "entities": dict(self.entities),
            "entity_provenance": {
                slot: item.to_dict()
                for slot, item in self.entity_provenance.items()
            },
            "missing_information": list(self.missing_information),
            "ambiguity": list(self.ambiguity),
            "clarification_required": self.clarification_required,
            "clarification_question": self.clarification_question,
            "relevant_references": list(self.relevant_references),
            "proposed_symbolic_task": (
                dict(self.proposed_symbolic_task)
                if self.proposed_symbolic_task is not None
                else None
            ),
            "unsupported_capabilities": list(self.unsupported_capabilities),
            "validation_status": self.validation_status,
            "dialogue_status": self.dialogue_status,
            "model_backend": self.model_backend,
            "model_errors": list(self.model_errors),
            "metadata": dict(self.metadata),
        }
