"""Constrained LLM-assisted planner.

Consumes READY DialogueResult (same gate as Phase 6), asks a provider for a
structured plan, parses it, grounds entities, and reuses Phase 6 validate_plan.
Never executes actions.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

from src.config_loader import load_llm_planning, load_planning
from src.dialogue.types import DialogueResult, DialogueStatus
from src.llm_planning.exceptions import (
    GroundingError,
    LLMPlanningInputError,
    MissingCredentialsError,
    ProviderError,
    ResponseParseError,
)
from src.llm_planning.grounding import assert_grounded
from src.llm_planning.prompt_builder import build_planning_prompt
from src.llm_planning.provider import LLMProvider, build_provider_from_config
from src.llm_planning.response_parser import parse_task_plan
from src.llm_planning.types import LLMPlanningResult, PlanningMode
from src.logging_utils import get_logger, setup_logging
from src.planning.exceptions import PlanValidationError
from src.planning.plan_validator import validate_plan
from src.planning.task_planner import TaskPlanner
from src.planning.types import PlanningStatus, TaskPlan
from src.validation.types import ValidationStatus

logger = get_logger("llm_planning")


@lru_cache(maxsize=1)
def _llm_config() -> dict:
    return load_llm_planning()


@lru_cache(maxsize=1)
def _planning_config() -> dict:
    return load_planning()


class LLMAssistedPlanner:
    """LLM-assisted planner with parse → ground → validate pipeline."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        *,
        deterministic_planner: Optional[TaskPlanner] = None,
        max_correction_attempts: Optional[int] = None,
        fallback_to_deterministic: Optional[bool] = None,
    ) -> None:
        setup_logging()
        config = _llm_config()
        self._provider = provider or build_provider_from_config(config)
        self._deterministic = deterministic_planner or TaskPlanner()
        self._max_corrections = (
            int(config.get("max_correction_attempts", 1))
            if max_correction_attempts is None
            else int(max_correction_attempts)
        )
        self._fallback = (
            bool(config.get("fallback_to_deterministic", False))
            if fallback_to_deterministic is None
            else bool(fallback_to_deterministic)
        )

    @property
    def provider_name(self) -> str:
        return getattr(self._provider, "name", type(self._provider).__name__)

    def plan(self, dialogue_result: DialogueResult) -> LLMPlanningResult:
        """Plan from a Phase 5 DialogueResult. Rejects non-READY dialogue."""
        if not isinstance(dialogue_result, DialogueResult):
            raise LLMPlanningInputError("plan() requires a DialogueResult.")

        if dialogue_result.status != DialogueStatus.READY:
            rejected = TaskPlan(
                status=PlanningStatus.REJECTED,
                intent=str(dialogue_result.task.get("intent", "")),
                entities=dict(dialogue_result.task.get("entities") or {}),
                errors=[
                    f"Dialogue status is {dialogue_result.status.value}; "
                    "only READY tasks may be planned."
                ],
                original_text=dialogue_result.original_text,
                explanation="LLM planning rejected: dialogue is not READY.",
            )
            return LLMPlanningResult(
                plan=rejected,
                mode=PlanningMode.LLM_ASSISTED,
                provider=self.provider_name,
                message="Rejected ineligible dialogue result.",
            )

        if dialogue_result.validation is not None:
            if dialogue_result.validation.status != ValidationStatus.VALID:
                rejected = TaskPlan(
                    status=PlanningStatus.REJECTED,
                    intent=str(dialogue_result.task.get("intent", "")),
                    entities=dict(dialogue_result.task.get("entities") or {}),
                    errors=[
                        "Attached ValidationResult is not VALID "
                        f"({dialogue_result.validation.status.value})."
                    ],
                    original_text=dialogue_result.original_text,
                    explanation="LLM planning rejected: validation is not VALID.",
                )
                return LLMPlanningResult(
                    plan=rejected,
                    mode=PlanningMode.LLM_ASSISTED,
                    provider=self.provider_name,
                    message="Rejected non-VALID validation.",
                )

        return self.plan_task(
            intent=str(dialogue_result.task.get("intent", "")),
            entities=dict(dialogue_result.task.get("entities") or {}),
            original_text=dialogue_result.original_text,
        )

    def plan_task(
        self,
        intent: str,
        entities: Dict[str, str],
        original_text: str = "",
    ) -> LLMPlanningResult:
        """Plan from an explicit intent + entities (tests / evaluation)."""
        plannable = {
            str(item) for item in _planning_config().get("plannable_intents") or []
        }
        if intent == "UNSUPPORTED" or intent not in plannable:
            plan = TaskPlan(
                status=PlanningStatus.UNSUPPORTED,
                intent=intent,
                entities=dict(entities),
                errors=[f"Intent '{intent}' is not plannable."],
                original_text=original_text,
                explanation="Unsupported or unknown intent for LLM planning.",
            )
            return LLMPlanningResult(
                plan=plan,
                mode=PlanningMode.LLM_ASSISTED,
                provider=self.provider_name,
                message="Intent not plannable.",
            )

        started = time.perf_counter()
        parse_errors: List[str] = []
        attempts = 0
        correction: Optional[Sequence[str]] = None
        max_attempts = 1 + max(0, self._max_corrections)

        while attempts < max_attempts:
            attempts += 1
            prompt = build_planning_prompt(
                intent=intent,
                entities=entities,
                original_text=original_text,
                correction_errors=correction,
            )
            try:
                raw = self._provider.complete(prompt)
            except MissingCredentialsError as exc:
                return self._fail_or_fallback(
                    intent=intent,
                    entities=entities,
                    original_text=original_text,
                    errors=[str(exc)],
                    parse_errors=parse_errors,
                    attempts=attempts,
                    started=started,
                    message="Missing provider credentials.",
                )
            except ProviderError as exc:
                return self._fail_or_fallback(
                    intent=intent,
                    entities=entities,
                    original_text=original_text,
                    errors=[str(exc)],
                    parse_errors=parse_errors,
                    attempts=attempts,
                    started=started,
                    message="Provider failure.",
                )

            try:
                plan = parse_task_plan(
                    raw,
                    fallback_intent=intent,
                    fallback_entities=entities,
                    original_text=original_text,
                )
                # Prefer caller grounding for intent/entities when model omits them.
                if not plan.entities:
                    plan = TaskPlan(
                        status=plan.status,
                        intent=plan.intent or intent,
                        entities=dict(entities),
                        actions=plan.actions,
                        errors=plan.errors,
                        warnings=plan.warnings,
                        original_text=original_text,
                        explanation=plan.explanation,
                    )
                if plan.status == PlanningStatus.PLANNED:
                    assert_grounded(plan)
                    validate_plan(plan)
                elif plan.actions:
                    # Non-PLANNED statuses should not carry executable actions.
                    raise PlanValidationError(
                        "Non-PLANNED status must not include executable actions."
                    )
            except (ResponseParseError, GroundingError, PlanValidationError) as exc:
                parse_errors.append(str(exc))
                correction = [str(exc)]
                logger.info(
                    "LLM plan attempt %d failed validation: %s",
                    attempts,
                    exc,
                )
                continue

            latency = (time.perf_counter() - started) * 1000.0
            logger.info(
                "LLM planned intent=%s actions=%d attempts=%d",
                plan.intent,
                len(plan.actions),
                attempts,
            )
            return LLMPlanningResult(
                plan=plan,
                mode=PlanningMode.LLM_ASSISTED,
                provider=self.provider_name,
                attempts=attempts,
                fallback_used=False,
                parse_errors=parse_errors,
                latency_ms=latency,
                message="LLM plan accepted after validation.",
            )

        return self._fail_or_fallback(
            intent=intent,
            entities=entities,
            original_text=original_text,
            errors=list(parse_errors),
            parse_errors=parse_errors,
            attempts=attempts,
            started=started,
            message="LLM plan failed validation after correction attempts.",
        )

    def _fail_or_fallback(
        self,
        *,
        intent: str,
        entities: Dict[str, str],
        original_text: str,
        errors: List[str],
        parse_errors: List[str],
        attempts: int,
        started: float,
        message: str,
    ) -> LLMPlanningResult:
        latency = (time.perf_counter() - started) * 1000.0
        if self._fallback:
            det = self._deterministic.plan_task(
                intent=intent,
                entities=entities,
                original_text=original_text,
            )
            # Only claim fallback success when deterministic planning succeeds.
            if det.status == PlanningStatus.PLANNED:
                warnings = list(det.warnings) + [
                    "Returned via deterministic fallback after LLM failure."
                ]
                plan = TaskPlan(
                    status=det.status,
                    intent=det.intent,
                    entities=det.entities,
                    actions=det.actions,
                    errors=det.errors,
                    warnings=warnings,
                    original_text=det.original_text,
                    explanation=det.explanation,
                )
                return LLMPlanningResult(
                    plan=plan,
                    mode=PlanningMode.DETERMINISTIC,
                    provider=self.provider_name,
                    attempts=attempts,
                    fallback_used=True,
                    parse_errors=parse_errors,
                    latency_ms=latency,
                    message=(
                        "LLM failed; deterministic planner produced a plan. "
                        + message
                    ),
                )
            # Deterministic also failed — surface combined errors.
            combined = list(errors) + list(det.errors)
            failed = TaskPlan(
                status=PlanningStatus.FAILED,
                intent=intent,
                entities=dict(entities),
                errors=combined,
                original_text=original_text,
                explanation="LLM and deterministic planners both failed.",
            )
            return LLMPlanningResult(
                plan=failed,
                mode=PlanningMode.LLM_ASSISTED,
                provider=self.provider_name,
                attempts=attempts,
                fallback_used=True,
                parse_errors=parse_errors,
                latency_ms=latency,
                message="LLM failed; deterministic fallback also failed.",
            )

        failed = TaskPlan(
            status=PlanningStatus.FAILED,
            intent=intent,
            entities=dict(entities),
            errors=list(errors),
            original_text=original_text,
            explanation=message,
        )
        return LLMPlanningResult(
            plan=failed,
            mode=PlanningMode.LLM_ASSISTED,
            provider=self.provider_name,
            attempts=attempts,
            fallback_used=False,
            parse_errors=parse_errors,
            latency_ms=latency,
            message=message,
        )
