"""Catalog grounding checks for LLM-proposed action parameters.

Phase 6 structural validation does not verify catalog membership.
Phase 7 adds this layer so the model cannot invent objects/locations/people.
"""

from __future__ import annotations

from typing import List

from src.llm_planning.exceptions import GroundingError
from src.planning.types import ActionType, TaskPlan
from src.validation.schema import load_policy

# Parameter name → catalog kind
_PARAM_CATALOG = {
    "object": "object",
    "destination": "location_or_user",
    "location": "location",
    "person": "person",
}


def grounding_errors(plan: TaskPlan) -> List[str]:
    """Return grounding problems; empty list means OK."""
    policy = load_policy()
    errors: List[str] = []

    for action in plan.actions:
        for key, value in action.parameters.items():
            kind = _PARAM_CATALOG.get(key)
            if kind is None:
                continue
            if kind == "object":
                if value not in policy.object_ids:
                    errors.append(
                        f"Action {action.id}: unknown object '{value}'."
                    )
            elif kind == "location":
                if value not in policy.location_ids:
                    errors.append(
                        f"Action {action.id}: unknown location '{value}'."
                    )
            elif kind == "location_or_user":
                allowed = set(policy.location_ids)
                if policy.allow_user_as_destination:
                    allowed.add(policy.user_value)
                if value not in allowed:
                    errors.append(
                        f"Action {action.id}: unknown destination '{value}'."
                    )
            elif kind == "person":
                if value not in policy.person_ids:
                    errors.append(
                        f"Action {action.id}: unknown person '{value}'."
                    )

        # NAVIGATE_TO_USER must not invent a destination parameter elsewhere.
        if action.type == ActionType.NAVIGATE_TO and not action.parameters.get(
            "destination"
        ):
            errors.append(
                f"Action {action.id}: NAVIGATE_TO requires destination."
            )
    return errors


def assert_grounded(plan: TaskPlan) -> None:
    errors = grounding_errors(plan)
    if errors:
        raise GroundingError("; ".join(errors))
