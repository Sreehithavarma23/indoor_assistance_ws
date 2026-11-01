"""Shared planning fixtures."""

from __future__ import annotations

import pytest

from src.dialogue import DialogueManager
from src.dialogue.types import DialogueResult, DialogueStatus
from src.nlu.dataset import load_train
from src.nlu.nlu import NLUEngine
from src.planning.task_planner import TaskPlanner
from src.validation.types import ValidationResult, ValidationStatus


@pytest.fixture(scope="session")
def nlu_engine():
    return NLUEngine.from_examples(load_train())


@pytest.fixture
def manager(nlu_engine):
    return DialogueManager(engine=nlu_engine)


@pytest.fixture
def planner():
    return TaskPlanner()


def make_ready(
    intent: str,
    entities: dict,
    *,
    original_text: str = "",
    validation_status: ValidationStatus = ValidationStatus.VALID,
) -> DialogueResult:
    """Build a synthetic READY DialogueResult for planner unit tests."""
    validation = ValidationResult(
        original_text=original_text,
        intent=intent,
        intent_confidence=1.0,
        entities=dict(entities),
        status=validation_status,
        reasons=[],
        missing_entities=[],
        ambiguous_entities=[],
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
    *,
    original_text: str = "",
) -> DialogueResult:
    return DialogueResult(
        status=status,
        question="Where?" if status == DialogueStatus.CLARIFICATION_REQUIRED else None,
        pending_slots=["source_location"]
        if status == DialogueStatus.CLARIFICATION_REQUIRED
        else [],
        task={
            "intent": intent,
            "entities": dict(entities or {}),
            "intent_confidence": 0.5,
            "original_text": original_text,
        },
        validation=None,
        clarification_count=1,
        original_text=original_text,
        message=status.value,
    )
