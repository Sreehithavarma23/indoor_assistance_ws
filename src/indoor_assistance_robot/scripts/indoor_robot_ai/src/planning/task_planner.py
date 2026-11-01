"""Deterministic high-level task planner.

Consumes a READY DialogueResult (or an explicit task dict) and emits a
TaskPlan of symbolic actions. Does not execute anything.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config_loader import load_planning
from src.dialogue.types import DialogueResult, DialogueStatus
from src.logging_utils import get_logger, setup_logging
from src.planning.exceptions import PlanningInputError, PlanValidationError
from src.planning.plan_validator import validate_plan
from src.planning.types import ActionType, PlannedAction, PlanningStatus, TaskPlan
from src.validation.types import ValidationStatus

logger = get_logger("planning")


@lru_cache(maxsize=1)
def _planning_config() -> dict:
    return load_planning()


def _action(
    index: int,
    action_type: ActionType,
    parameters: Optional[Dict[str, str]] = None,
    depends_on: Optional[Sequence[str]] = None,
    description: str = "",
) -> PlannedAction:
    action_id = f"action_{index}"
    return PlannedAction(
        id=action_id,
        type=action_type,
        parameters=dict(parameters or {}),
        order=index,
        depends_on=list(depends_on or []),
        description=description,
    )


def _chain(actions: List[PlannedAction]) -> List[PlannedAction]:
    """Each action depends on the previous one (sequential plan)."""
    chained: List[PlannedAction] = []
    for index, action in enumerate(actions, start=1):
        deps = [actions[index - 2].id] if index > 1 else []
        chained.append(
            PlannedAction(
                id=f"action_{index}",
                type=action.type,
                parameters=dict(action.parameters),
                order=index,
                depends_on=deps,
                description=action.description,
            )
        )
    return chained


class TaskPlanner:
    """Map validated READY tasks to deterministic high-level plans."""

    def __init__(self) -> None:
        setup_logging()

    def plan(self, dialogue_result: DialogueResult) -> TaskPlan:
        """Plan from a Phase 5 DialogueResult. Rejects non-READY dialogue."""
        if not isinstance(dialogue_result, DialogueResult):
            raise PlanningInputError("plan() requires a DialogueResult.")

        if dialogue_result.status != DialogueStatus.READY:
            return TaskPlan(
                status=PlanningStatus.REJECTED,
                intent=str(dialogue_result.task.get("intent", "")),
                entities=dict(dialogue_result.task.get("entities") or {}),
                errors=[
                    f"Dialogue status is {dialogue_result.status.value}; "
                    "only READY tasks may be planned."
                ],
                original_text=dialogue_result.original_text,
                explanation="Planning rejected: dialogue is not READY.",
            )

        if dialogue_result.validation is not None:
            if dialogue_result.validation.status != ValidationStatus.VALID:
                return TaskPlan(
                    status=PlanningStatus.REJECTED,
                    intent=str(dialogue_result.task.get("intent", "")),
                    entities=dict(dialogue_result.task.get("entities") or {}),
                    errors=[
                        "Attached ValidationResult is not VALID "
                        f"({dialogue_result.validation.status.value})."
                    ],
                    original_text=dialogue_result.original_text,
                    explanation="Planning rejected: validation is not VALID.",
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
    ) -> TaskPlan:
        """Plan from an explicit intent + entity mapping (for tests/eval)."""
        config = _planning_config()
        plannable = {str(item) for item in config.get("plannable_intents") or []}
        user_value = str(config.get("user_destination_value", "user"))

        if intent == "UNSUPPORTED" or intent not in plannable:
            return TaskPlan(
                status=PlanningStatus.UNSUPPORTED,
                intent=intent,
                entities=dict(entities),
                errors=[f"Intent '{intent}' is not plannable."],
                original_text=original_text,
                explanation="Unsupported or unknown intent for planning.",
            )

        try:
            actions, warnings, explanation = self._build_actions(
                intent, entities, user_value
            )
        except PlanningInputError as exc:
            return TaskPlan(
                status=PlanningStatus.FAILED,
                intent=intent,
                entities=dict(entities),
                errors=[str(exc)],
                original_text=original_text,
                explanation="Planning failed: required entities missing.",
            )

        plan = TaskPlan(
            status=PlanningStatus.PLANNED,
            intent=intent,
            entities=dict(entities),
            actions=actions,
            warnings=warnings,
            original_text=original_text,
            explanation=explanation,
        )
        try:
            validate_plan(plan)
        except PlanValidationError as exc:
            return TaskPlan(
                status=PlanningStatus.FAILED,
                intent=intent,
                entities=dict(entities),
                errors=[str(exc)],
                original_text=original_text,
                explanation="Generated plan failed structural validation.",
            )

        logger.info(
            "Planned intent=%s actions=%d",
            intent,
            len(plan.actions),
        )
        return plan

    def _require(self, entities: Dict[str, str], key: str) -> str:
        value = entities.get(key)
        if not value:
            raise PlanningInputError(f"Missing required entity '{key}' for planning.")
        return value

    def _build_actions(
        self,
        intent: str,
        entities: Dict[str, str],
        user_value: str,
    ) -> Tuple[List[PlannedAction], List[str], str]:
        warnings: List[str] = []

        if intent == "NAVIGATE":
            destination = self._require(entities, "destination")
            if destination == user_value:
                actions = [
                    _action(
                        1,
                        ActionType.NAVIGATE_TO_USER,
                        description="Navigate to the requesting user.",
                    )
                ]
            else:
                actions = [
                    _action(
                        1,
                        ActionType.NAVIGATE_TO,
                        {"destination": destination},
                        description=f"Navigate to {destination}.",
                    )
                ]
            return actions, warnings, f"NAVIGATE → go to {destination}."

        if intent == "RETURN":
            destination = self._require(entities, "destination")
            actions = [
                _action(
                    1,
                    ActionType.NAVIGATE_TO,
                    {"destination": destination},
                    description=f"Return / navigate to {destination}.",
                )
            ]
            return actions, warnings, f"RETURN → navigate to {destination}."

        if intent == "FOLLOW":
            person = self._require(entities, "person")
            actions = [
                _action(
                    1,
                    ActionType.FOLLOW_PERSON,
                    {"person": person},
                    description=f"Follow {person}.",
                )
            ]
            return actions, warnings, f"FOLLOW → follow {person}."

        if intent == "STOP":
            actions = [
                _action(1, ActionType.STOP, description="Symbolic stop request.")
            ]
            return actions, warnings, "STOP → stop current activity."

        if intent == "RETRIEVE":
            obj = self._require(entities, "object")
            source = self._require(entities, "source_location")
            destination = self._require(entities, "destination")
            draft = [
                _action(
                    1,
                    ActionType.NAVIGATE_TO,
                    {"destination": source},
                    description=f"Go to {source}.",
                ),
                _action(
                    2,
                    ActionType.SEARCH_OBJECT,
                    {"object": obj, "location": source},
                    description=f"Search for {obj} at {source}.",
                ),
                _action(
                    3,
                    ActionType.APPROACH_OBJECT,
                    {"object": obj},
                    description=f"Approach {obj}.",
                ),
                _action(
                    4,
                    ActionType.GRASP_OBJECT,
                    {"object": obj},
                    description=f"Grasp {obj}.",
                ),
            ]
            if destination == user_value:
                draft.append(
                    _action(
                        5,
                        ActionType.NAVIGATE_TO_USER,
                        description="Return to the user.",
                    )
                )
                draft.append(
                    _action(
                        6,
                        ActionType.DELIVER_OBJECT,
                        {"object": obj, "destination": user_value},
                        description=f"Deliver {obj} to the user.",
                    )
                )
            else:
                draft.append(
                    _action(
                        5,
                        ActionType.NAVIGATE_TO,
                        {"destination": destination},
                        description=f"Go to {destination}.",
                    )
                )
                draft.append(
                    _action(
                        6,
                        ActionType.DELIVER_OBJECT,
                        {"object": obj, "destination": destination},
                        description=f"Deliver {obj} to {destination}.",
                    )
                )
            return (
                _chain(draft),
                warnings,
                f"RETRIEVE → fetch {obj} from {source} to {destination}.",
            )

        if intent == "DELIVER":
            obj = self._require(entities, "object")
            person = entities.get("person")
            destination = entities.get("destination")
            if not person and not destination:
                raise PlanningInputError(
                    "DELIVER requires person or destination."
                )
            if _planning_config().get("deliver_assumes_object_held", True):
                warnings.append(
                    "DELIVER assumes the object is already held; "
                    "grasp/possession state is not tracked in Phases 1–5."
                )
            draft: List[PlannedAction] = []
            if person:
                draft.append(
                    _action(
                        1,
                        ActionType.NAVIGATE_TO_PERSON,
                        {"person": person},
                        description=f"Go to {person}.",
                    )
                )
                draft.append(
                    _action(
                        2,
                        ActionType.DELIVER_OBJECT,
                        {"object": obj, "person": person},
                        description=f"Deliver {obj} to {person}.",
                    )
                )
            elif destination == user_value:
                draft.append(
                    _action(
                        1,
                        ActionType.NAVIGATE_TO_USER,
                        description="Go to the user.",
                    )
                )
                draft.append(
                    _action(
                        2,
                        ActionType.DELIVER_OBJECT,
                        {"object": obj, "destination": user_value},
                        description=f"Deliver {obj} to the user.",
                    )
                )
            else:
                assert destination is not None
                draft.append(
                    _action(
                        1,
                        ActionType.NAVIGATE_TO,
                        {"destination": destination},
                        description=f"Go to {destination}.",
                    )
                )
                draft.append(
                    _action(
                        2,
                        ActionType.DELIVER_OBJECT,
                        {"object": obj, "destination": destination},
                        description=f"Deliver {obj} to {destination}.",
                    )
                )
            return (
                _chain(draft),
                warnings,
                f"DELIVER → deliver {obj} "
                f"({'person=' + person if person else 'destination=' + str(destination)}).",
            )

        raise PlanningInputError(f"No planning rule for intent '{intent}'.")
