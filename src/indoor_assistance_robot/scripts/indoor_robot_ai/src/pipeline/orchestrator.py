"""Unified AI pipeline orchestrator (Phases 2–7).

Coordinates existing modules. Does not reimplement NLU, validation, dialogue,
or planning algorithms. Does not execute robot actions.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, Union

from src.config_loader import load_pipeline
from src.dialogue.types import DialogueResult, DialogueStatus
from src.language_understanding import from_dialogue_result
from src.llm_planning.planner import LLMAssistedPlanner
from src.llm_planning.provider import LLMProvider
from src.llm_planning.types import PlanningMode
from src.logging_utils import get_logger, setup_logging
from src.nlu.nlu import NLUEngine
from src.pipeline.session import PipelineSession
from src.pipeline.types import PipelineResult, PipelineStatus, StageTimings
from src.planning.task_planner import TaskPlanner
from src.planning.types import PlanningStatus, TaskPlan
from src.speech.pipeline import transcribe_file
from src.speech.types import Transcript

logger = get_logger("pipeline")

# Injectable ASR boundary for tests: (path) -> Transcript
TranscribeFn = Callable[[Union[str, Path]], Transcript]


@lru_cache(maxsize=1)
def _pipeline_config() -> dict:
    return load_pipeline()


def _ms_since(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _dialogue_to_pipeline_status(status: DialogueStatus) -> PipelineStatus:
    mapping = {
        DialogueStatus.CLARIFICATION_REQUIRED: PipelineStatus.CLARIFICATION_REQUIRED,
        DialogueStatus.CLARIFICATION_FAILED: PipelineStatus.CLARIFICATION_FAILED,
        DialogueStatus.CANCELLED: PipelineStatus.CANCELLED,
        DialogueStatus.UNSUPPORTED: PipelineStatus.UNSUPPORTED,
    }
    return mapping.get(status, PipelineStatus.FAILED)


class AIPipeline:
    """End-to-end speech/text → dialogue → symbolic TaskPlan orchestrator."""

    def __init__(
        self,
        *,
        engine: Optional[NLUEngine] = None,
        deterministic_planner: Optional[TaskPlanner] = None,
        llm_planner: Optional[LLMAssistedPlanner] = None,
        llm_provider: Optional[LLMProvider] = None,
        transcribe_fn: Optional[TranscribeFn] = None,
        default_mode: str = "deterministic",
    ) -> None:
        setup_logging()
        config = _pipeline_config()
        self.engine = engine
        self.deterministic_planner = deterministic_planner or TaskPlanner()
        if llm_planner is not None:
            self.llm_planner = llm_planner
        else:
            self.llm_planner = LLMAssistedPlanner(provider=llm_provider)
        self.transcribe_fn = transcribe_fn or transcribe_file
        self.default_mode = str(
            default_mode or config.get("default_planning_mode") or "deterministic"
        ).lower()
        self.reject_empty_transcript = bool(
            config.get("reject_empty_transcript", True)
        )

    def new_session(self) -> PipelineSession:
        """Create an isolated conversation session."""
        return PipelineSession(engine=self.engine)

    def run_text(
        self,
        text: str,
        *,
        session: Optional[PipelineSession] = None,
        mode: Optional[str] = None,
    ) -> PipelineResult:
        """Process one natural-language text turn."""
        total_started = time.perf_counter()
        if text is None or not isinstance(text, str) or not text.strip():
            return PipelineResult(
                status=PipelineStatus.FAILED,
                input_text="" if text is None else str(text),
                input_source="text",
                errors=["Text input is empty or invalid."],
                message="Invalid text input.",
                timings=StageTimings(total_ms=_ms_since(total_started)),
            )

        active = session or self.new_session()
        try:
            planning_mode = self._normalize_mode(mode)
        except ValueError as exc:
            return PipelineResult(
                status=PipelineStatus.FAILED,
                input_text=text.strip(),
                input_source="text",
                errors=[str(exc)],
                message="Invalid planning mode.",
                timings=StageTimings(total_ms=_ms_since(total_started)),
            )

        dialogue_started = time.perf_counter()
        try:
            dialogue = active.dialogue.process(text.strip())
        except Exception as exc:  # surface as FAILED; do not invent success
            logger.exception("Dialogue processing failed")
            return PipelineResult(
                status=PipelineStatus.FAILED,
                input_text=text.strip(),
                input_source="text",
                errors=[str(exc)],
                message="Dialogue processing failed.",
                timings=StageTimings(
                    dialogue_ms=_ms_since(dialogue_started),
                    total_ms=_ms_since(total_started),
                ),
            )
        dialogue_ms = _ms_since(dialogue_started)

        return self._after_dialogue(
            dialogue=dialogue,
            input_text=text.strip(),
            input_source="text",
            transcript=None,
            planning_mode=planning_mode,
            dialogue_ms=dialogue_ms,
            asr_ms=None,
            total_started=total_started,
        )

    def run_audio(
        self,
        audio_path: Union[str, Path],
        *,
        session: Optional[PipelineSession] = None,
        mode: Optional[str] = None,
    ) -> PipelineResult:
        """Transcribe an audio file, then run the same text pipeline."""
        total_started = time.perf_counter()
        path = Path(audio_path)
        asr_started = time.perf_counter()
        try:
            transcript = self.transcribe_fn(path)
        except Exception as exc:
            logger.exception("ASR failed for %s", path)
            return PipelineResult(
                status=PipelineStatus.FAILED,
                input_text="",
                input_source="audio",
                errors=[str(exc)],
                message="Speech processing failed.",
                timings=StageTimings(
                    asr_ms=_ms_since(asr_started),
                    total_ms=_ms_since(total_started),
                ),
            )
        asr_ms = _ms_since(asr_started)
        text = (transcript.text or "").strip()

        if self.reject_empty_transcript and not text:
            return PipelineResult(
                status=PipelineStatus.FAILED,
                input_text="",
                input_source="audio",
                transcript=transcript,
                errors=["Empty transcript; no speech text available for NLU."],
                message="ASR produced an empty transcript.",
                timings=StageTimings(
                    asr_ms=asr_ms,
                    total_ms=_ms_since(total_started),
                ),
            )

        active = session or self.new_session()
        try:
            planning_mode = self._normalize_mode(mode)
        except ValueError as exc:
            return PipelineResult(
                status=PipelineStatus.FAILED,
                input_text=text,
                input_source="audio",
                transcript=transcript,
                errors=[str(exc)],
                message="Invalid planning mode.",
                timings=StageTimings(
                    asr_ms=asr_ms,
                    total_ms=_ms_since(total_started),
                ),
            )

        dialogue_started = time.perf_counter()
        try:
            dialogue = active.dialogue.process(text)
        except Exception as exc:
            logger.exception("Dialogue processing failed after ASR")
            return PipelineResult(
                status=PipelineStatus.FAILED,
                input_text=text,
                input_source="audio",
                transcript=transcript,
                errors=[str(exc)],
                message="Dialogue processing failed.",
                timings=StageTimings(
                    asr_ms=asr_ms,
                    dialogue_ms=_ms_since(dialogue_started),
                    total_ms=_ms_since(total_started),
                ),
            )
        dialogue_ms = _ms_since(dialogue_started)

        return self._after_dialogue(
            dialogue=dialogue,
            input_text=text,
            input_source="audio",
            transcript=transcript,
            planning_mode=planning_mode,
            dialogue_ms=dialogue_ms,
            asr_ms=asr_ms,
            total_started=total_started,
        )

    def _normalize_mode(self, mode: Optional[str]) -> str:
        value = (mode or self.default_mode or "deterministic").strip().lower()
        if value in {"deterministic", "det"}:
            return "deterministic"
        if value in {"llm", "llm_assisted"}:
            return "llm"
        raise ValueError(
            f"Unknown planning mode '{mode}'. Use 'deterministic' or 'llm'."
        )

    def _after_dialogue(
        self,
        *,
        dialogue: DialogueResult,
        input_text: str,
        input_source: str,
        transcript: Optional[Transcript],
        planning_mode: str,
        dialogue_ms: float,
        asr_ms: Optional[float],
        total_started: float,
    ) -> PipelineResult:
        intent = None
        entities: dict = {}
        intent_confidence = None
        if dialogue.task:
            intent = dialogue.task.get("intent")
            entities = dict(dialogue.task.get("entities") or {})
            confidence = dialogue.task.get("intent_confidence")
            if isinstance(confidence, (int, float)):
                intent_confidence = float(confidence)

        validation_status = None
        if dialogue.validation is not None:
            validation_status = dialogue.validation.status.value

        base_kwargs = dict(
            input_text=input_text,
            input_source=input_source,
            transcript=transcript,
            dialogue_status=dialogue.status.value,
            intent=intent if isinstance(intent, str) else None,
            entities=entities,
            language_understanding=from_dialogue_result(dialogue),
            intent_confidence=intent_confidence,
            validation_status=validation_status,
            clarification_question=dialogue.question,
            pending_slots=list(dialogue.pending_slots),
            message=dialogue.message or "",
        )

        if dialogue.status != DialogueStatus.READY:
            return PipelineResult(
                status=_dialogue_to_pipeline_status(dialogue.status),
                planning_mode=None,
                planning_status=None,
                plan=None,
                errors=[],
                warnings=[],
                timings=StageTimings(
                    asr_ms=asr_ms,
                    dialogue_ms=dialogue_ms,
                    planning_ms=None,
                    total_ms=_ms_since(total_started),
                ),
                **base_kwargs,
            )

        # READY → plan with the explicitly selected mode only.
        plan_started = time.perf_counter()
        selected = (
            PlanningMode.DETERMINISTIC
            if planning_mode == "deterministic"
            else PlanningMode.LLM_ASSISTED
        )
        try:
            if planning_mode == "deterministic":
                plan = self.deterministic_planner.plan(dialogue)
                provider = None
                fallback_used = False
                warnings = list(plan.warnings)
                errors = list(plan.errors)
            else:
                llm_result = self.llm_planner.plan(dialogue)
                plan = llm_result.plan
                selected = llm_result.mode
                provider = llm_result.provider
                fallback_used = llm_result.fallback_used
                warnings = list(plan.warnings)
                errors = list(plan.errors) + list(llm_result.parse_errors)
                if llm_result.message:
                    base_kwargs["message"] = llm_result.message
        except Exception as exc:
            logger.exception("Planning failed")
            return PipelineResult(
                status=PipelineStatus.FAILED,
                planning_mode=selected,
                planning_status=PlanningStatus.FAILED,
                plan=None,
                errors=[str(exc)],
                message="Planning failed.",
                timings=StageTimings(
                    asr_ms=asr_ms,
                    dialogue_ms=dialogue_ms,
                    planning_ms=_ms_since(plan_started),
                    total_ms=_ms_since(total_started),
                ),
                **base_kwargs,
            )

        planning_ms = _ms_since(plan_started)
        pipeline_status = self._status_from_plan(plan)
        return PipelineResult(
            status=pipeline_status,
            planning_mode=selected,
            planning_status=plan.status,
            plan=plan,
            planner_provider=provider,
            fallback_used=fallback_used,
            errors=errors,
            warnings=warnings,
            timings=StageTimings(
                asr_ms=asr_ms,
                dialogue_ms=dialogue_ms,
                planning_ms=planning_ms,
                total_ms=_ms_since(total_started),
            ),
            **base_kwargs,
        )

    @staticmethod
    def _status_from_plan(plan: TaskPlan) -> PipelineStatus:
        if plan.status == PlanningStatus.PLANNED:
            return PipelineStatus.PLANNED
        if plan.status == PlanningStatus.UNSUPPORTED:
            return PipelineStatus.UNSUPPORTED
        if plan.status == PlanningStatus.REJECTED:
            return PipelineStatus.REJECTED
        return PipelineStatus.FAILED
