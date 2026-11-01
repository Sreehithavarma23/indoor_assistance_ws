"""Structured high-level plan types.

These are symbolic specifications for a future execution layer.
They are not motor commands and do not imply physical success.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ActionType(str, Enum):
    """Supported symbolic high-level actions."""

    NAVIGATE_TO = "NAVIGATE_TO"
    NAVIGATE_TO_USER = "NAVIGATE_TO_USER"
    NAVIGATE_TO_PERSON = "NAVIGATE_TO_PERSON"
    SEARCH_OBJECT = "SEARCH_OBJECT"
    APPROACH_OBJECT = "APPROACH_OBJECT"
    GRASP_OBJECT = "GRASP_OBJECT"
    DELIVER_OBJECT = "DELIVER_OBJECT"
    FOLLOW_PERSON = "FOLLOW_PERSON"
    STOP = "STOP"


class PlanningStatus(str, Enum):
    """Outcome of planning. Not robot execution success."""

    PLANNED = "PLANNED"
    REJECTED = "REJECTED"
    UNSUPPORTED = "UNSUPPORTED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class PlannedAction:
    """One ordered symbolic action in a TaskPlan."""

    id: str
    type: ActionType
    parameters: Dict[str, str] = field(default_factory=dict)
    order: int = 0
    depends_on: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "parameters": dict(self.parameters),
            "order": self.order,
            "depends_on": list(self.depends_on),
            "description": self.description,
        }


@dataclass(frozen=True)
class TaskPlan:
    """Deterministic high-level plan for a validated READY task."""

    status: PlanningStatus
    intent: str
    entities: Dict[str, str]
    actions: List[PlannedAction] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    original_text: str = ""
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "intent": self.intent,
            "entities": dict(self.entities),
            "actions": [action.to_dict() for action in self.actions],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "original_text": self.original_text,
            "explanation": self.explanation,
        }
