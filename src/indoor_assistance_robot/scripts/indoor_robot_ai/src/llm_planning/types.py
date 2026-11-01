"""Result types for constrained LLM-assisted planning.

Reuses Phase 6 TaskPlan / PlannedAction. Adds planning-mode metadata so
LLM vs deterministic outcomes are never confused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.planning.types import TaskPlan


class PlanningMode(str, Enum):
    """Which planner produced the accepted (or final) plan."""

    DETERMINISTIC = "DETERMINISTIC"
    LLM_ASSISTED = "LLM_ASSISTED"


@dataclass(frozen=True)
class LLMPlanningResult:
    """Outcome of an LLM-assisted planning attempt.

    ``plan`` is always a Phase 6 TaskPlan. Metadata records how it was obtained.
    """

    plan: TaskPlan
    mode: PlanningMode
    provider: str
    attempts: int = 0
    fallback_used: bool = False
    parse_errors: List[str] = field(default_factory=list)
    latency_ms: Optional[float] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "mode": self.mode.value,
            "provider": self.provider,
            "attempts": self.attempts,
            "fallback_used": self.fallback_used,
            "parse_errors": list(self.parse_errors),
            "latency_ms": self.latency_ms,
            "message": self.message,
        }
