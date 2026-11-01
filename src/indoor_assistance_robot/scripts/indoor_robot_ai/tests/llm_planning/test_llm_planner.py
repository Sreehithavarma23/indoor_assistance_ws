"""Unit and integration tests for Phase 7 LLM-assisted planning."""

from __future__ import annotations

import json

import pytest

from src.dialogue.types import DialogueStatus
from src.llm_planning.exceptions import (
    MissingCredentialsError,
    ProviderError,
    ProviderTimeoutError,
)
from src.llm_planning.planner import LLMAssistedPlanner
from src.llm_planning.provider import FakeLLMProvider, OpenAICompatibleProvider
from src.llm_planning.response_parser import parse_task_plan
from src.llm_planning.types import PlanningMode
from src.planning.types import ActionType, PlanningStatus
from tests.llm_planning.conftest import make_dialogue, make_ready, navigate_plan_json


class TestValidLLMPlan:
    def test_valid_navigate_plan(self):
        provider = FakeLLMProvider([navigate_plan_json()])
        result = LLMAssistedPlanner(
            provider=provider,
            max_correction_attempts=0,
            fallback_to_deterministic=False,
        ).plan(make_ready("NAVIGATE", {"destination": "kitchen"}))
        assert result.plan.status == PlanningStatus.PLANNED
        assert result.mode == PlanningMode.LLM_ASSISTED
        assert result.fallback_used is False
        assert result.attempts == 1
        assert result.plan.actions[0].type == ActionType.NAVIGATE_TO
        assert result.plan.actions[0].parameters["destination"] == "kitchen"

    def test_dialogue_integration(self, manager):
        dialogue = manager.process("Take me to the kitchen.")
        assert dialogue.status == DialogueStatus.READY
        provider = FakeLLMProvider([navigate_plan_json()])
        result = LLMAssistedPlanner(
            provider=provider,
            fallback_to_deterministic=False,
        ).plan(dialogue)
        assert result.plan.status == PlanningStatus.PLANNED
        assert result.plan.actions[0].parameters["destination"] == "kitchen"


class TestEligibility:
    @pytest.mark.parametrize(
        "status",
        [
            DialogueStatus.CLARIFICATION_REQUIRED,
            DialogueStatus.CLARIFICATION_FAILED,
            DialogueStatus.CANCELLED,
            DialogueStatus.UNSUPPORTED,
        ],
    )
    def test_rejects_non_ready(self, status):
        provider = FakeLLMProvider([navigate_plan_json()])
        result = LLMAssistedPlanner(provider=provider).plan(
            make_dialogue(status, "NAVIGATE", {"destination": "kitchen"})
        )
        assert result.plan.status == PlanningStatus.REJECTED
        assert provider.prompts == []  # must not call the model

    def test_incomplete_dialogue_path(self, manager):
        dialogue = manager.process("Bring me the bottle.")
        assert dialogue.status == DialogueStatus.CLARIFICATION_REQUIRED
        provider = FakeLLMProvider([navigate_plan_json()])
        result = LLMAssistedPlanner(provider=provider).plan(dialogue)
        assert result.plan.status == PlanningStatus.REJECTED
        assert provider.prompts == []


class TestInvalidModelOutput:
    def test_malformed_json(self):
        provider = FakeLLMProvider(["not-json{"])
        result = LLMAssistedPlanner(
            provider=provider,
            max_correction_attempts=0,
            fallback_to_deterministic=False,
        ).plan_task("NAVIGATE", {"destination": "kitchen"})
        assert result.plan.status == PlanningStatus.FAILED
        assert result.parse_errors

    def test_unexpected_action_type(self):
        bad = navigate_plan_json(action_type="FLY_TO")
        # navigate_plan_json with FLY_TO still puts destination in params only for NAVIGATE_TO
        payload = json.loads(navigate_plan_json())
        payload["actions"][0]["type"] = "FLY_TO"
        provider = FakeLLMProvider([json.dumps(payload)])
        result = LLMAssistedPlanner(
            provider=provider,
            max_correction_attempts=0,
            fallback_to_deterministic=False,
        ).plan_task("NAVIGATE", {"destination": "kitchen"})
        assert result.plan.status == PlanningStatus.FAILED

    def test_missing_required_parameter(self):
        payload = json.loads(navigate_plan_json())
        payload["actions"][0]["parameters"] = {}
        provider = FakeLLMProvider([json.dumps(payload)])
        result = LLMAssistedPlanner(
            provider=provider,
            max_correction_attempts=0,
            fallback_to_deterministic=False,
        ).plan_task("NAVIGATE", {"destination": "kitchen"})
        assert result.plan.status == PlanningStatus.FAILED

    def test_unknown_entity_rejected(self):
        payload = json.loads(navigate_plan_json(destination="atlantis"))
        provider = FakeLLMProvider([json.dumps(payload)])
        result = LLMAssistedPlanner(
            provider=provider,
            max_correction_attempts=0,
            fallback_to_deterministic=False,
        ).plan_task("NAVIGATE", {"destination": "kitchen"})
        assert result.plan.status == PlanningStatus.FAILED
        assert any("unknown destination" in e.lower() for e in result.parse_errors)

    def test_duplicate_ids(self):
        payload = {
            "status": "PLANNED",
            "intent": "FOLLOW",
            "entities": {"person": "john"},
            "actions": [
                {
                    "id": "a1",
                    "type": "FOLLOW_PERSON",
                    "parameters": {"person": "john"},
                    "order": 1,
                    "depends_on": [],
                },
                {
                    "id": "a1",
                    "type": "FOLLOW_PERSON",
                    "parameters": {"person": "john"},
                    "order": 2,
                    "depends_on": [],
                },
            ],
            "explanation": "bad",
            "errors": [],
            "warnings": [],
        }
        provider = FakeLLMProvider([json.dumps(payload)])
        result = LLMAssistedPlanner(
            provider=provider,
            max_correction_attempts=0,
            fallback_to_deterministic=False,
        ).plan_task("FOLLOW", {"person": "john"})
        assert result.plan.status == PlanningStatus.FAILED

    def test_dependency_cycle(self):
        payload = {
            "status": "PLANNED",
            "intent": "FOLLOW",
            "entities": {"person": "john"},
            "actions": [
                {
                    "id": "a1",
                    "type": "FOLLOW_PERSON",
                    "parameters": {"person": "john"},
                    "order": 1,
                    "depends_on": ["a2"],
                },
                {
                    "id": "a2",
                    "type": "FOLLOW_PERSON",
                    "parameters": {"person": "john"},
                    "order": 2,
                    "depends_on": ["a1"],
                },
            ],
            "explanation": "bad",
            "errors": [],
            "warnings": [],
        }
        provider = FakeLLMProvider([json.dumps(payload)])
        result = LLMAssistedPlanner(
            provider=provider,
            max_correction_attempts=0,
            fallback_to_deterministic=False,
        ).plan_task("FOLLOW", {"person": "john"})
        assert result.plan.status == PlanningStatus.FAILED

    def test_self_dependency(self):
        payload = {
            "status": "PLANNED",
            "intent": "STOP",
            "entities": {},
            "actions": [
                {
                    "id": "a1",
                    "type": "STOP",
                    "parameters": {},
                    "order": 1,
                    "depends_on": ["a1"],
                }
            ],
            "explanation": "bad",
            "errors": [],
            "warnings": [],
        }
        provider = FakeLLMProvider([json.dumps(payload)])
        result = LLMAssistedPlanner(
            provider=provider,
            max_correction_attempts=0,
            fallback_to_deterministic=False,
        ).plan_task("STOP", {})
        assert result.plan.status == PlanningStatus.FAILED


class TestProviderFailures:
    def test_missing_credentials(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = OpenAICompatibleProvider(
            api_base_url="https://example.invalid/v1",
            model="test",
            api_key_env="OPENAI_API_KEY",
        )
        with pytest.raises(MissingCredentialsError):
            provider.complete("hello")

    def test_provider_error_surfaces_as_failed_plan(self):
        provider = FakeLLMProvider(fail_with=ProviderError("boom"))
        result = LLMAssistedPlanner(
            provider=provider,
            max_correction_attempts=0,
            fallback_to_deterministic=False,
        ).plan_task("NAVIGATE", {"destination": "kitchen"})
        assert result.plan.status == PlanningStatus.FAILED
        assert "boom" in result.plan.errors[0]

    def test_timeout_error(self):
        provider = FakeLLMProvider(fail_with=ProviderTimeoutError("timed out"))
        result = LLMAssistedPlanner(
            provider=provider,
            fallback_to_deterministic=False,
        ).plan_task("STOP", {})
        assert result.plan.status == PlanningStatus.FAILED

    def test_bounded_correction_then_accept(self):
        bad = "not-json"
        good = navigate_plan_json()
        provider = FakeLLMProvider([bad, good])
        result = LLMAssistedPlanner(
            provider=provider,
            max_correction_attempts=1,
            fallback_to_deterministic=False,
        ).plan_task("NAVIGATE", {"destination": "kitchen"})
        assert result.plan.status == PlanningStatus.PLANNED
        assert result.attempts == 2
        assert len(provider.prompts) == 2
        assert "previous_validation_errors" in provider.prompts[1]

    def test_repeated_invalid_exhausted(self):
        provider = FakeLLMProvider(["{", "{"])
        result = LLMAssistedPlanner(
            provider=provider,
            max_correction_attempts=1,
            fallback_to_deterministic=False,
        ).plan_task("NAVIGATE", {"destination": "kitchen"})
        assert result.plan.status == PlanningStatus.FAILED
        assert result.attempts == 2


class TestFallback:
    def test_fallback_uses_deterministic(self):
        provider = FakeLLMProvider(["not-json"])
        result = LLMAssistedPlanner(
            provider=provider,
            max_correction_attempts=0,
            fallback_to_deterministic=True,
        ).plan_task("NAVIGATE", {"destination": "kitchen"})
        assert result.plan.status == PlanningStatus.PLANNED
        assert result.fallback_used is True
        assert result.mode == PlanningMode.DETERMINISTIC
        assert result.plan.actions[0].type == ActionType.NAVIGATE_TO
        assert any("fallback" in w.lower() for w in result.plan.warnings)

    def test_no_silent_fallback_when_disabled(self):
        provider = FakeLLMProvider(["not-json"])
        result = LLMAssistedPlanner(
            provider=provider,
            max_correction_attempts=0,
            fallback_to_deterministic=False,
        ).plan_task("NAVIGATE", {"destination": "kitchen"})
        assert result.fallback_used is False
        assert result.plan.status == PlanningStatus.FAILED


class TestParser:
    def test_accepts_fenced_json(self):
        body = navigate_plan_json()
        fenced = f"```json\n{body}\n```"
        plan = parse_task_plan(
            fenced,
            fallback_intent="NAVIGATE",
            fallback_entities={"destination": "kitchen"},
        )
        assert plan.status == PlanningStatus.PLANNED
        assert plan.actions[0].type == ActionType.NAVIGATE_TO


class TestDeterminismOfFakeProvider:
    def test_repeatable(self):
        response = navigate_plan_json()
        first = LLMAssistedPlanner(
            provider=FakeLLMProvider([response]),
            fallback_to_deterministic=False,
        ).plan_task("NAVIGATE", {"destination": "kitchen"})
        second = LLMAssistedPlanner(
            provider=FakeLLMProvider([response]),
            fallback_to_deterministic=False,
        ).plan_task("NAVIGATE", {"destination": "kitchen"})
        assert first.plan.to_dict() == second.plan.to_dict()
