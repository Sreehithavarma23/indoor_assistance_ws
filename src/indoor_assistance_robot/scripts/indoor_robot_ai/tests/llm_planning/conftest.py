"""Shared fixtures for Phase 7 LLM planning tests."""

from __future__ import annotations

import json

import pytest

from src.dialogue import DialogueManager
from src.dialogue.types import DialogueResult, DialogueStatus
from src.nlu.dataset import load_train
from src.nlu.nlu import NLUEngine
from src.validation.types import ValidationResult, ValidationStatus


@pytest.fixture(scope="session")
def nlu_engine():
    return NLUEngine.from_examples(load_train())


@pytest.fixture
def manager(nlu_engine):
    return DialogueManager(engine=nlu_engine)


def make_ready(
    intent: str,
    entities: dict,
    *,
    original_text: str = "",
    validation_status: ValidationStatus = ValidationStatus.VALID,
) -> DialogueResult:
    validation = ValidationResult(
        original_text=original_text,
        intent=intent,
        intent_confidence=1.0,
        entities=dict(entities),
        status=validation_status,
    )
    return DialogueResult(
        status=DialogueStatus.READY,
        question=None,
        pending_slots=[],
        task={
            "intent": intent,
            "entities": dict(entities),
            "intent_confidence": 1.0,
            "original_text": original_text,
        },
        validation=validation,
        clarification_count=0,
        original_text=original_text,
    )


def make_dialogue(
    status: DialogueStatus,
    intent: str = "",
    entities: dict | None = None,
) -> DialogueResult:
    return DialogueResult(
        status=status,
        question=None,
        pending_slots=[],
        task={
            "intent": intent,
            "entities": dict(entities or {}),
            "intent_confidence": 0.5,
            "original_text": "",
        },
        validation=None,
        clarification_count=0,
        original_text="",
        message=status.value,
    )


def navigate_plan_json(
    destination: str = "kitchen",
    *,
    action_type: str = "NAVIGATE_TO",
) -> str:
    return json.dumps(
        {
            "status": "PLANNED",
            "intent": "NAVIGATE",
            "entities": {"destination": destination},
            "actions": [
                {
                    "id": "action_1",
                    "type": action_type,
                    "parameters": {"destination": destination}
                    if action_type == "NAVIGATE_TO"
                    else {},
                    "order": 1,
                    "depends_on": [],
                    "description": f"Go to {destination}.",
                }
            ],
            "explanation": f"Navigate to {destination}.",
            "errors": [],
            "warnings": [],
        }
    )
