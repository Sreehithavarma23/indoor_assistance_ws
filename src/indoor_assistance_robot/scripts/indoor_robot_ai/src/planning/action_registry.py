"""Explicit vocabulary of symbolic high-level actions.

Each action is a planning token for a future ROS 2 / Nav2 / MoveIt layer.
None of these actions are executed in Phase 6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from src.planning.types import ActionType


@dataclass(frozen=True)
class ActionSpec:
    """Schema for one supported action type."""

    type: ActionType
    required_parameters: List[str]
    optional_parameters: List[str]
    purpose: str
    capability_note: str


ACTION_REGISTRY: Dict[ActionType, ActionSpec] = {
    ActionType.NAVIGATE_TO: ActionSpec(
        type=ActionType.NAVIGATE_TO,
        required_parameters=["destination"],
        optional_parameters=[],
        purpose="Move the mobile base to a named map location.",
        capability_note="Requires Member 2 Nav2 / location waypoints.",
    ),
    ActionType.NAVIGATE_TO_USER: ActionSpec(
        type=ActionType.NAVIGATE_TO_USER,
        required_parameters=[],
        optional_parameters=[],
        purpose="Move the base to the requesting user.",
        capability_note="Requires user localization; not implemented in Phase 6.",
    ),
    ActionType.NAVIGATE_TO_PERSON: ActionSpec(
        type=ActionType.NAVIGATE_TO_PERSON,
        required_parameters=["person"],
        optional_parameters=[],
        purpose="Move the base to a named person.",
        capability_note="Requires person tracking; not implemented in Phase 6.",
    ),
    ActionType.SEARCH_OBJECT: ActionSpec(
        type=ActionType.SEARCH_OBJECT,
        required_parameters=["object", "location"],
        optional_parameters=[],
        purpose="Search for a named object at a named location.",
        capability_note="Requires Member 3 vision / detection.",
    ),
    ActionType.APPROACH_OBJECT: ActionSpec(
        type=ActionType.APPROACH_OBJECT,
        required_parameters=["object"],
        optional_parameters=[],
        purpose="Approach a detected object for manipulation.",
        capability_note="Requires detection pose + base/arm positioning.",
    ),
    ActionType.GRASP_OBJECT: ActionSpec(
        type=ActionType.GRASP_OBJECT,
        required_parameters=["object"],
        optional_parameters=[],
        purpose="Grasp a named object with the manipulator.",
        capability_note="Requires Member 3 MoveIt / gripper.",
    ),
    ActionType.DELIVER_OBJECT: ActionSpec(
        type=ActionType.DELIVER_OBJECT,
        required_parameters=["object"],
        optional_parameters=["destination", "person"],
        purpose="Hand over or place a grasped object.",
        capability_note="Requires grasp state + place/handoff skill.",
    ),
    ActionType.FOLLOW_PERSON: ActionSpec(
        type=ActionType.FOLLOW_PERSON,
        required_parameters=["person"],
        optional_parameters=[],
        purpose="Follow a named person.",
        capability_note="Requires person tracking + following controller.",
    ),
    ActionType.STOP: ActionSpec(
        type=ActionType.STOP,
        required_parameters=[],
        optional_parameters=[],
        purpose="Symbolic request to stop current motion.",
        capability_note="Safety stop belongs to future execution, not this planner.",
    ),
}


def supported_action_types() -> Set[str]:
    return {action.value for action in ACTION_REGISTRY}


def get_action_spec(action_type: ActionType) -> ActionSpec:
    return ACTION_REGISTRY[action_type]
