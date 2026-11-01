"""Action registry coverage."""

from __future__ import annotations

from src.planning.action_registry import ACTION_REGISTRY, supported_action_types
from src.planning.types import ActionType


def test_all_action_types_registered():
    for action_type in ActionType:
        assert action_type in ACTION_REGISTRY


def test_supported_names():
    names = supported_action_types()
    assert "NAVIGATE_TO" in names
    assert "GRASP_OBJECT" in names
    assert "STOP" in names
