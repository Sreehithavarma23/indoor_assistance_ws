"""End-to-end tests for the Phase 8 unified AI pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.llm_planning.exceptions import ProviderError
from src.llm_planning.planner import LLMAssistedPlanner
from src.llm_planning.provider import FakeLLMProvider
from src.llm_planning.types import PlanningMode
from src.pipeline import AIPipeline, PipelineSession
from src.pipeline.types import PipelineStatus
from src.planning.task_planner import TaskPlanner
from src.planning.types import ActionType, PlanningStatus
from src.speech.types import Transcript


def _nav_json() -> str:
    return json.dumps(
        {
            "status": "PLANNED",
            "intent": "NAVIGATE",
            "entities": {"destination": "kitchen"},
            "actions": [
                {
                    "id": "action_1",
                    "type": "NAVIGATE_TO",
                    "parameters": {"destination": "kitchen"},
                    "order": 1,
                    "depends_on": [],
                }
            ],
            "explanation": "nav",
            "errors": [],
            "warnings": [],
        }
    )


class TestValidCommands:
    def test_navigate(self, pipeline):
        result = pipeline.run_text("Take me to the kitchen.", mode="deterministic")
        assert result.status == PipelineStatus.PLANNED
        assert result.intent == "NAVIGATE"
        assert result.dialogue_status == "READY"
        assert result.planning_mode == PlanningMode.DETERMINISTIC
        assert result.plan is not None
        assert result.plan.actions[0].type == ActionType.NAVIGATE_TO
        assert result.plan.actions[0].parameters["destination"] == "kitchen"
        assert result.timings.asr_ms is None
        assert result.timings.planning_ms is not None

    def test_retrieve(self, pipeline):
        result = pipeline.run_text(
            "Bring me the bottle from the kitchen.", mode="deterministic"
        )
        assert result.status == PipelineStatus.PLANNED
        assert result.intent == "RETRIEVE"
        assert result.entities.get("object") == "bottle"
        assert result.entities.get("source_location") == "kitchen"
        types = [a.type for a in result.plan.actions]
        assert types[0] == ActionType.NAVIGATE_TO
        assert types[-1] == ActionType.DELIVER_OBJECT
        for index, action in enumerate(result.plan.actions):
            if index == 0:
                assert action.depends_on == []
            else:
                assert action.depends_on == [result.plan.actions[index - 1].id]

    def test_stop(self, pipeline):
        result = pipeline.run_text("Stop.", mode="deterministic")
        assert result.status == PipelineStatus.PLANNED
        assert result.intent == "STOP"
        assert result.plan.actions[0].type == ActionType.STOP


class TestClarificationAndUnsupported:
    @pytest.mark.parametrize(
        "text",
        [
            "Take me to the kitchen or the office.",
            "Bring me the bottle or the cup.",
        ],
    )
    def test_alternative_required_entities_never_reach_planner(self, pipeline, text):
        det = MagicMock(wraps=TaskPlanner())
        pipe = AIPipeline(engine=pipeline.engine, deterministic_planner=det)
        result = pipe.run_text(text, mode="deterministic")

        assert result.status == PipelineStatus.CLARIFICATION_REQUIRED
        assert result.plan is None
        assert result.planning_status is None
        det.plan.assert_not_called()

    def test_missing_information_no_plan(self, pipeline):
        det = MagicMock(wraps=TaskPlanner())
        pipe = AIPipeline(engine=pipeline.engine, deterministic_planner=det)
        result = pipe.run_text("Bring me the bottle.", mode="deterministic")
        assert result.status == PipelineStatus.CLARIFICATION_REQUIRED
        assert result.clarification_question
        assert "source_location" in result.pending_slots
        assert result.planning_status is None
        det.plan.assert_not_called()

    def test_follow_up_clarification(self, pipeline):
        det = MagicMock(wraps=TaskPlanner())
        pipe = AIPipeline(engine=pipeline.engine, deterministic_planner=det)
        session = pipe.new_session()
        first = pipe.run_text(
            "Bring me the bottle.", session=session, mode="deterministic"
        )
        assert first.status == PipelineStatus.CLARIFICATION_REQUIRED
        det.plan.assert_not_called()
        second = pipe.run_text(
            "From the kitchen.", session=session, mode="deterministic"
        )
        assert second.status == PipelineStatus.PLANNED
        assert second.intent == "RETRIEVE"
        assert second.entities.get("source_location") == "kitchen"
        det.plan.assert_called_once()

    def test_unsupported_no_planner(self, pipeline):
        det = MagicMock(wraps=TaskPlanner())
        pipe = AIPipeline(engine=pipeline.engine, deterministic_planner=det)
        result = pipe.run_text("What's the weather today?", mode="deterministic")
        assert result.status == PipelineStatus.UNSUPPORTED
        det.plan.assert_not_called()

    def test_go_there_no_plan(self, pipeline):
        det = MagicMock(wraps=TaskPlanner())
        pipe = AIPipeline(engine=pipeline.engine, deterministic_planner=det)
        result = pipe.run_text("Go there.", mode="deterministic")
        assert result.status == PipelineStatus.CLARIFICATION_REQUIRED
        assert result.intent == "NAVIGATE"
        assert result.plan is None or result.planning_status != PlanningStatus.PLANNED
        det.plan.assert_not_called()


class TestLLMMode:
    def test_llm_path_selected(self, nlu_engine):
        provider = FakeLLMProvider([_nav_json()])
        llm = LLMAssistedPlanner(
            provider=provider,
            max_correction_attempts=0,
            fallback_to_deterministic=False,
        )
        det = MagicMock(wraps=TaskPlanner())
        pipe = AIPipeline(
            engine=nlu_engine,
            deterministic_planner=det,
            llm_planner=llm,
        )
        result = pipe.run_text("Take me to the kitchen.", mode="llm")
        assert result.status == PipelineStatus.PLANNED
        assert result.planning_mode == PlanningMode.LLM_ASSISTED
        assert result.planner_provider == "fake"
        assert provider.prompts  # provider was called
        det.plan.assert_not_called()

    def test_invalid_llm_output_rejected(self, nlu_engine):
        bad = json.dumps(
            {
                "status": "PLANNED",
                "intent": "NAVIGATE",
                "entities": {"destination": "kitchen"},
                "actions": [
                    {
                        "id": "a1",
                        "type": "FLY_TO",
                        "parameters": {"destination": "kitchen"},
                        "order": 1,
                        "depends_on": [],
                    }
                ],
                "explanation": "bad",
                "errors": [],
                "warnings": [],
            }
        )
        provider = FakeLLMProvider([bad])
        pipe = AIPipeline(
            engine=nlu_engine,
            llm_planner=LLMAssistedPlanner(
                provider=provider,
                max_correction_attempts=0,
                fallback_to_deterministic=False,
            ),
        )
        result = pipe.run_text("Take me to the kitchen.", mode="llm")
        assert result.status == PipelineStatus.FAILED
        assert result.planning_status != PlanningStatus.PLANNED
        assert result.fallback_used is False

    def test_provider_failure_explicit(self, nlu_engine):
        provider = FakeLLMProvider(fail_with=ProviderError("unavailable"))
        pipe = AIPipeline(
            engine=nlu_engine,
            llm_planner=LLMAssistedPlanner(
                provider=provider,
                max_correction_attempts=0,
                fallback_to_deterministic=False,
            ),
        )
        result = pipe.run_text("Take me to the kitchen.", mode="llm")
        assert result.status == PipelineStatus.FAILED
        assert result.fallback_used is False
        assert result.planning_mode == PlanningMode.LLM_ASSISTED


class TestSessionIsolation:
    def test_sessions_do_not_share_state(self, pipeline):
        s1 = pipeline.new_session()
        s2 = pipeline.new_session()
        r1 = pipeline.run_text("Bring me the bottle.", session=s1)
        assert r1.status == PipelineStatus.CLARIFICATION_REQUIRED
        r2 = pipeline.run_text("Take me to the kitchen.", session=s2)
        assert r2.status == PipelineStatus.PLANNED
        assert r2.intent == "NAVIGATE"
        # s1 still awaiting; follow-up on s1 should merge, not use s2 state
        r3 = pipeline.run_text("From the kitchen.", session=s1)
        assert r3.status == PipelineStatus.PLANNED
        assert r3.intent == "RETRIEVE"

    def test_reset_clears_pending(self, pipeline):
        session = pipeline.new_session()
        pipeline.run_text("Bring me the bottle.", session=session)
        session.reset()
        result = pipeline.run_text("From the kitchen.", session=session)
        # Without pending retrieve, bare "From the kitchen." is not a retrieval plan.
        assert result.status != PipelineStatus.PLANNED or result.intent != "RETRIEVE"


class TestAudioPath:
    def test_audio_uses_transcript_text(self, nlu_engine):
        transcript = Transcript(
            text="Take me to the kitchen.",
            speech_detected=True,
            source="file",
            audio_path="fake.wav",
        )
        asr = MagicMock(return_value=transcript)
        det = MagicMock(wraps=TaskPlanner())
        pipe = AIPipeline(
            engine=nlu_engine,
            deterministic_planner=det,
            transcribe_fn=asr,
        )
        result = pipe.run_audio("fake.wav", mode="deterministic")
        asr.assert_called_once()
        assert result.input_source == "audio"
        assert result.input_text == "Take me to the kitchen."
        assert result.status == PipelineStatus.PLANNED
        assert result.timings.asr_ms is not None
        det.plan.assert_called_once()

    def test_empty_transcript_fails(self, nlu_engine):
        transcript = Transcript(text="", speech_detected=False, source="file")
        pipe = AIPipeline(
            engine=nlu_engine,
            transcribe_fn=lambda _p: transcript,
        )
        result = pipe.run_audio("silent.wav")
        assert result.status == PipelineStatus.FAILED
        assert "Empty transcript" in result.errors[0]


class TestSerializationAndInput:
    def test_planned_serializes(self, pipeline):
        result = pipeline.run_text("Stop.")
        payload = result.to_dict()
        assert payload["status"] == "PLANNED"
        assert isinstance(payload["plan"]["actions"][0]["type"], str)
        json.dumps(payload)  # must be JSON-safe

    def test_clarification_serializes(self, pipeline):
        result = pipeline.run_text("Bring me the bottle.")
        payload = result.to_dict()
        assert payload["status"] == "CLARIFICATION_REQUIRED"
        assert payload["plan"] is None
        json.dumps(payload)

    def test_empty_text(self, pipeline):
        result = pipeline.run_text("   ")
        assert result.status == PipelineStatus.FAILED

    def test_conflicting_cli_args(self):
        from src.pipeline.__main__ import main

        with pytest.raises(SystemExit) as exc:
            main(["--text", "hi", "--audio", "x.wav"])
        assert exc.value.code == 2


class TestNoPlannerGuard:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Bring me the bottle.", PipelineStatus.CLARIFICATION_REQUIRED),
            ("What's the weather today?", PipelineStatus.UNSUPPORTED),
            ("Go there.", PipelineStatus.CLARIFICATION_REQUIRED),
        ],
    )
    def test_non_ready_never_calls_planners(self, nlu_engine, text, expected):
        det = MagicMock(wraps=TaskPlanner())
        llm_inner = LLMAssistedPlanner(
            provider=FakeLLMProvider([_nav_json()]),
            fallback_to_deterministic=False,
        )
        llm = MagicMock(wraps=llm_inner)
        pipe = AIPipeline(
            engine=nlu_engine,
            deterministic_planner=det,
            llm_planner=llm,
        )
        result = pipe.run_text(text, mode="deterministic")
        assert result.status == expected
        det.plan.assert_not_called()
        llm.plan.assert_not_called()
