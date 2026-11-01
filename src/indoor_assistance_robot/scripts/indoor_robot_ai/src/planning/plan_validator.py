"""Structural validation of generated TaskPlan objects.

Checks plan representation only — not physical feasibility.
"""

from __future__ import annotations

from typing import List

from src.planning.action_registry import ACTION_REGISTRY, get_action_spec
from src.planning.exceptions import PlanValidationError
from src.planning.types import PlannedAction, TaskPlan


def _validate_action(action: PlannedAction, known_ids: set[str]) -> List[str]:
    errors: List[str] = []
    if action.type not in ACTION_REGISTRY:
        errors.append(f"Unsupported action type: {action.type}")
        return errors

    spec = get_action_spec(action.type)
    for param in spec.required_parameters:
        value = action.parameters.get(param)
        if not value or not isinstance(value, str):
            errors.append(
                f"Action {action.id} ({action.type.value}) missing required "
                f"parameter '{param}'."
            )
    for key, value in action.parameters.items():
        if not isinstance(key, str) or not isinstance(value, str):
            errors.append(
                f"Action {action.id} parameters must be string→string; "
                f"got {key!r}={value!r}."
            )
        allowed = set(spec.required_parameters) | set(spec.optional_parameters)
        if key not in allowed:
            errors.append(
                f"Action {action.id} has unexpected parameter '{key}'."
            )

    if action.id in action.depends_on:
        errors.append(f"Action {action.id} depends on itself.")

    for dep in action.depends_on:
        if dep not in known_ids:
            errors.append(
                f"Action {action.id} depends on unknown action '{dep}'."
            )
    return errors


def _has_cycle(actions: List[PlannedAction]) -> bool:
    graph = {action.id: list(action.depends_on) for action in actions}
    visited: set[str] = set()
    stack: set[str] = set()

    def dfs(node: str) -> bool:
        if node in stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        stack.add(node)
        for nxt in graph.get(node, []):
            if dfs(nxt):
                return True
        stack.remove(node)
        return False

    return any(dfs(action.id) for action in actions)


def _order_respects_dependencies(actions: List[PlannedAction]) -> List[str]:
    """For sequential plans, each dependency must have a smaller order."""
    by_id = {action.id: action for action in actions}
    errors: List[str] = []
    for action in actions:
        for dep in action.depends_on:
            prerequisite = by_id.get(dep)
            if prerequisite is not None and prerequisite.order >= action.order:
                errors.append(
                    f"Action {action.id} order={action.order} depends on "
                    f"{dep} order={prerequisite.order}; dependency must precede."
                )
    return errors


def validate_plan(plan: TaskPlan) -> None:
    """Raise PlanValidationError if the plan structure is invalid."""
    errors: List[str] = []
    ids = [action.id for action in plan.actions]
    if len(ids) != len(set(ids)):
        errors.append("Action identifiers must be unique.")

    known = set(ids)
    for action in plan.actions:
        errors.extend(_validate_action(action, known))

    if _has_cycle(plan.actions):
        errors.append("Action dependency graph contains a cycle.")

    errors.extend(_order_respects_dependencies(plan.actions))

    if errors:
        raise PlanValidationError("; ".join(errors))
