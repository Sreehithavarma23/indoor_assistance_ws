"""Parse and structurally coerce LLM text into a Phase 6 TaskPlan."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from src.llm_planning.exceptions import ResponseParseError
from src.planning.types import ActionType, PlannedAction, PlanningStatus, TaskPlan


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract a JSON object from raw model text."""
    if not isinstance(text, str) or not text.strip():
        raise ResponseParseError("Empty model response.")

    candidates = [text.strip()]
    match = _FENCE_RE.search(text)
    if match:
        candidates.insert(0, match.group(1).strip())

    # Also try substring from first { to last }.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(data, dict):
            return data
        last_error = ResponseParseError("JSON root must be an object.")
    raise ResponseParseError(
        f"Malformed JSON in model response: {last_error}"
    )


def _parse_status(raw: Any) -> PlanningStatus:
    if raw is None:
        return PlanningStatus.PLANNED
    if not isinstance(raw, str):
        raise ResponseParseError("status must be a string.")
    try:
        return PlanningStatus(raw)
    except ValueError as exc:
        raise ResponseParseError(f"Unknown planning status '{raw}'.") from exc


def _parse_action(item: Any, index: int) -> PlannedAction:
    if not isinstance(item, dict):
        raise ResponseParseError(f"Action at index {index} must be an object.")
    action_id = item.get("id")
    if not isinstance(action_id, str) or not action_id.strip():
        raise ResponseParseError(f"Action at index {index} missing string id.")
    type_raw = item.get("type")
    if not isinstance(type_raw, str):
        raise ResponseParseError(
            f"Action {action_id} missing string type."
        )
    try:
        action_type = ActionType(type_raw)
    except ValueError as exc:
        raise ResponseParseError(
            f"Unsupported action type '{type_raw}'."
        ) from exc

    parameters = item.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ResponseParseError(
            f"Action {action_id} parameters must be an object."
        )
    clean_params: Dict[str, str] = {}
    for key, value in parameters.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ResponseParseError(
                f"Action {action_id} parameters must be string→string."
            )
        clean_params[key] = value

    depends_on = item.get("depends_on") or []
    if not isinstance(depends_on, list) or not all(
        isinstance(dep, str) for dep in depends_on
    ):
        raise ResponseParseError(
            f"Action {action_id} depends_on must be a list of strings."
        )

    order = item.get("order", index + 1)
    if not isinstance(order, int) or isinstance(order, bool):
        raise ResponseParseError(
            f"Action {action_id} order must be an integer."
        )

    description = item.get("description") or ""
    if not isinstance(description, str):
        raise ResponseParseError(
            f"Action {action_id} description must be a string."
        )

    return PlannedAction(
        id=action_id,
        type=action_type,
        parameters=clean_params,
        order=order,
        depends_on=list(depends_on),
        description=description,
    )


def parse_task_plan(
    text: str,
    *,
    fallback_intent: str,
    fallback_entities: Dict[str, str],
    original_text: str = "",
) -> TaskPlan:
    """Parse model text into a TaskPlan (structure only; validate separately)."""
    data = extract_json_object(text)
    status = _parse_status(data.get("status"))
    intent = data.get("intent", fallback_intent)
    if not isinstance(intent, str):
        raise ResponseParseError("intent must be a string.")

    entities = data.get("entities", fallback_entities)
    if entities is None:
        entities = fallback_entities
    if not isinstance(entities, dict):
        raise ResponseParseError("entities must be an object.")
    clean_entities: Dict[str, str] = {}
    for key, value in entities.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ResponseParseError("entities must be string→string.")
        clean_entities[key] = value

    actions_raw = data.get("actions") or []
    if not isinstance(actions_raw, list):
        raise ResponseParseError("actions must be a list.")
    actions: List[PlannedAction] = [
        _parse_action(item, index) for index, item in enumerate(actions_raw)
    ]

    errors = data.get("errors") or []
    warnings = data.get("warnings") or []
    if not isinstance(errors, list) or not all(isinstance(x, str) for x in errors):
        raise ResponseParseError("errors must be a list of strings.")
    if not isinstance(warnings, list) or not all(
        isinstance(x, str) for x in warnings
    ):
        raise ResponseParseError("warnings must be a list of strings.")

    explanation = data.get("explanation") or ""
    if not isinstance(explanation, str):
        raise ResponseParseError("explanation must be a string.")

    return TaskPlan(
        status=status,
        intent=intent,
        entities=clean_entities,
        actions=actions,
        errors=list(errors),
        warnings=list(warnings),
        original_text=original_text or str(data.get("original_text") or ""),
        explanation=explanation,
    )
