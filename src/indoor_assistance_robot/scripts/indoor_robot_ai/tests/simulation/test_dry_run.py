"""Dry-run adapter must never claim live execution."""

from __future__ import annotations

import pytest

from src.pipeline.types import PipelineResult, PipelineStatus
from src.planning.types import ActionType, PlannedAction, PlanningStatus, TaskPlan
from src.speech.types import Transcript
from src.simulation.dry_run import DryRunAdapter, load_capability_map


def test_capability_map_loads():
    data = load_capability_map()
    assert "action_bindings" in data
    assert data["action_bindings"]["GRASP_OBJECT"]["status"] == "unsupported_in_sim"
    assert data["interfaces"]["grasp_action"]["verified"] is False


def test_live_ros_flag_rejected():
    with pytest.raises(ValueError, match="Live ROS"):
        DryRunAdapter(allow_live_ros=True)


def test_navigate_dry_run_does_not_execute():
    adapter = DryRunAdapter()
    result = adapter.run_text("Take me to the kitchen.")
    assert result.would_execute is False
    assert result.success is True
    assert result.adapter_status == "DRY_RUN_ACCEPTED"
    assert result.rejection_reason
    assert result.intent == "NAVIGATE"
    assert result.pipeline_status in {"PLANNED", "CLARIFICATION_REQUIRED", "FAILED"}


def test_unsupported_dry_run():
    adapter = DryRunAdapter()
    result = adapter.run_text("What is the weather today?")
    assert result.would_execute is False
    assert result.pipeline_status in {"UNSUPPORTED", "CLARIFICATION_REQUIRED", "FAILED", "REJECTED"}


@pytest.mark.parametrize(
    "action",
    [
        PlannedAction(
            id="action_1",
            type=ActionType.NAVIGATE_TO,
            parameters={"destination": "mars"},
            order=1,
        ),
        PlannedAction(
            id="action_1",
            type=ActionType.GRASP_OBJECT,
            parameters={},
            order=1,
        ),
    ],
)
def test_malformed_or_unknown_plan_is_rejected(tmp_path, action):
    plan = TaskPlan(
        status=PlanningStatus.PLANNED,
        intent="NAVIGATE",
        entities={},
        actions=[action],
    )

    class FakePipeline:
        def run_text(self, text, mode="deterministic"):
            return PipelineResult(
                status=PipelineStatus.PLANNED,
                input_text=text,
                input_source="text",
                plan=plan,
            )

    adapter = DryRunAdapter(pipeline=FakePipeline(), log_dir=tmp_path)
    result = adapter.run_text("navigate")
    assert result.success is False
    assert result.adapter_status == "REJECTED"
    assert result.intended_operations == []
    assert result.would_execute is False


def test_empty_audio_transcript_is_rejected_without_plan(tmp_path):
    class FakePipeline:
        def run_audio(self, audio_path, mode="deterministic"):
            return PipelineResult(
                status=PipelineStatus.FAILED,
                input_text="",
                input_source="audio",
                transcript=Transcript(text="", speech_detected=False),
                errors=["Empty transcript"],
            )

    adapter = DryRunAdapter(pipeline=FakePipeline(), log_dir=tmp_path)
    result = adapter.run_audio("silent.wav")
    assert result.success is False
    assert result.adapter_status == "REJECTED"
    assert result.plan_actions == []
    assert result.would_execute is False
