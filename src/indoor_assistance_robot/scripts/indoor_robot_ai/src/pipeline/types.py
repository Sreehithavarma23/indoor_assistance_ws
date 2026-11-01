"""Serializable end-to-end pipeline result for one turn."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.language_understanding.types import LanguageUnderstandingResult
from src.llm_planning.types import PlanningMode
from src.planning.types import PlanningStatus, TaskPlan
from src.speech.types import Transcript


class PipelineStatus(str, Enum):
    """High-level outcome of one pipeline turn. Not robot execution."""

    PLANNED = "PLANNED"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    CLARIFICATION_FAILED = "CLARIFICATION_FAILED"
    CANCELLED = "CANCELLED"
    UNSUPPORTED = "UNSUPPORTED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class StageTimings:
    """Elapsed milliseconds for stages that actually ran. Monotonic clock."""

    asr_ms: Optional[float] = None
    dialogue_ms: Optional[float] = None
    planning_ms: Optional[float] = None
    total_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asr_ms": self.asr_ms,
            "dialogue_ms": self.dialogue_ms,
            "planning_ms": self.planning_ms,
            "total_ms": self.total_ms,
        }


@dataclass(frozen=True)
class PipelineResult:
    """One invocation of the unified AI pipeline.

    Reuses Phase 5/6 structures. Does not invent confidence or success.
    """

    status: PipelineStatus
    input_text: str
    input_source: str  # "text" | "audio" | "transcript"
    transcript: Optional[Transcript] = None
    dialogue_status: Optional[str] = None
    intent: Optional[str] = None
    entities: Dict[str, str] = field(default_factory=dict)
    language_understanding: Optional[LanguageUnderstandingResult] = None
    intent_confidence: Optional[float] = None
    validation_status: Optional[str] = None
    clarification_question: Optional[str] = None
    pending_slots: List[str] = field(default_factory=list)
    planning_mode: Optional[PlanningMode] = None
    planning_status: Optional[PlanningStatus] = None
    plan: Optional[TaskPlan] = None
    planner_provider: Optional[str] = None
    fallback_used: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    message: str = ""
    timings: StageTimings = field(default_factory=StageTimings)

    def to_dict(self) -> Dict[str, Any]:
        transcript_payload = None
        if self.transcript is not None:
            transcript_payload = {
                "text": self.transcript.text,
                "language": self.transcript.language,
                "speech_detected": self.transcript.speech_detected,
                "source": self.transcript.source,
                "audio_path": self.transcript.audio_path,
                "sample_rate": self.transcript.sample_rate,
                "latency_seconds": self.transcript.latency_seconds,
                "pipeline_latency_seconds": self.transcript.pipeline_latency_seconds,
            }
        return {
            "status": self.status.value,
            "input_text": self.input_text,
            "input_source": self.input_source,
            "transcript": transcript_payload,
            "dialogue_status": self.dialogue_status,
            "intent": self.intent,
            "entities": dict(self.entities),
            "language_understanding": (
                self.language_understanding.to_dict()
                if self.language_understanding
                else None
            ),
            "intent_confidence": self.intent_confidence,
            "validation_status": self.validation_status,
            "clarification_question": self.clarification_question,
            "pending_slots": list(self.pending_slots),
            "planning_mode": self.planning_mode.value if self.planning_mode else None,
            "planning_status": (
                self.planning_status.value if self.planning_status else None
            ),
            "plan": self.plan.to_dict() if self.plan else None,
            "planner_provider": self.planner_provider,
            "fallback_used": self.fallback_used,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "message": self.message,
            "timings": self.timings.to_dict(),
        }
