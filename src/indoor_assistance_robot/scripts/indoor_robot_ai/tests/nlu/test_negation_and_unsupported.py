"""Regression tests for negated movement and unsupported capability recovery."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.dialogue import DialogueManager
from src.dialogue.types import DialogueStatus
from src.nlu.cues import has_negated_movement, matches_unsupported_capability
from src.nlu.normalization import normalize_text
from src.pipeline import AIPipeline
from src.pipeline.types import PipelineStatus
from src.planning.task_planner import TaskPlanner
from src.validation import validate
from src.validation.types import ValidationStatus


@pytest.mark.parametrize(
    "text",
    [
        "Don't go any further.",
        "Don’t go any further.",  # curly apostrophe
        "Do not go any further.",
        "Don't move.",
        "Stop moving.",
    ],
)
def test_negated_movement_final_intent_is_stop(nlu_engine, text):
    result = nlu_engine.parse(text)
    assert result.intent == "STOP"
    assert result.entities.get("action") != "navigate"


def test_positive_go_there_remains_navigate(nlu_engine):
    result = nlu_engine.parse("Go there.")
    assert result.intent == "NAVIGATE"
    assert result.entities.get("action") == "navigate"


def test_go_grab_medicine_is_retrieve_not_navigate_action(nlu_engine):
    result = nlu_engine.parse("Go grab the medicine from the bedroom.")
    assert result.intent == "RETRIEVE"
    assert result.entities.get("action") == "retrieve"
    assert result.entities.get("object") == "medicine"
    assert result.entities.get("source_location") == "bedroom"


def test_dont_bring_is_not_treated_as_negated_movement():
    """Negation of non-movement verbs must not blindly become STOP."""
    assert not has_negated_movement("Don't bring the bottle.")


@pytest.mark.parametrize(
    "text",
    [
        "Could you fly me to the roof?",
        "Please compose a poem.",
        "Switch off the air conditioner.",
        "Can you hack the wifi?",
        "Make me a sandwich from scratch.",
        "Paint the wall blue.",
        "Recite the national anthem.",
        "Do a backflip.",
        "Start a video call with dad.",
    ],
)
def test_unsupported_capability_examples(nlu_engine, text):
    result = nlu_engine.parse(text)
    assert result.intent == "UNSUPPORTED"
    assert matches_unsupported_capability(text)


def test_apostrophe_normalization_keeps_dont_together():
    assert normalize_text("Don't go.") == "dont go"
    assert normalize_text("Don’t go.") == "dont go"
    assert has_negated_movement("Don’t go any further.")


def test_mixed_positive_and_negated_movement_is_contradictory(nlu_engine):
    from src.nlu.cues import has_contradictory_instruction

    text = "Go to the kitchen and don't go to the kitchen."
    assert has_contradictory_instruction(text)
    assert nlu_engine.parse(text).intent == "UNSUPPORTED"


def test_contradictory_movement_never_reaches_planner(nlu_engine):
    det = MagicMock(wraps=TaskPlanner())
    pipeline = AIPipeline(engine=nlu_engine, deterministic_planner=det)
    result = pipeline.run_text("Go to the kitchen and don't go to the kitchen.")
    assert result.status == PipelineStatus.UNSUPPORTED
    assert result.plan is None
    det.plan.assert_not_called()


def test_negated_movement_is_ready_stop_and_plans(nlu_engine):
    """STOP after negation is READY; planner may run. Incomplete must not."""
    det = MagicMock(wraps=TaskPlanner())
    pipe = AIPipeline(engine=nlu_engine, deterministic_planner=det)
    result = pipe.run_text("Don't go any further.", mode="deterministic")
    assert result.status == PipelineStatus.PLANNED
    assert result.intent == "STOP"
    assert result.dialogue_status == "READY"
    det.plan.assert_called_once()


def test_incomplete_request_does_not_invoke_planner(nlu_engine):
    det = MagicMock(wraps=TaskPlanner())
    pipe = AIPipeline(engine=nlu_engine, deterministic_planner=det)
    result = pipe.run_text("Bring me the bottle.", mode="deterministic")
    assert result.status == PipelineStatus.CLARIFICATION_REQUIRED
    assert result.dialogue_status != "READY"
    det.plan.assert_not_called()


def test_unsupported_request_does_not_invoke_planner(nlu_engine):
    det = MagicMock(wraps=TaskPlanner())
    pipe = AIPipeline(engine=nlu_engine, deterministic_planner=det)
    result = pipe.run_text("Can you hack the wifi?", mode="deterministic")
    assert result.status == PipelineStatus.UNSUPPORTED
    assert result.plan is None
    assert result.dialogue_status != "READY"
    det.plan.assert_not_called()


def test_unsupported_validation_even_if_intent_mislabeled():
    from tests.validation.conftest import make_nlu

    nlu = make_nlu("Paint the wall blue.", "RETRIEVE", {})
    result = validate(nlu)
    assert result.status == ValidationStatus.UNSUPPORTED


@pytest.mark.parametrize(
    "text",
    [
        "Go to Narnia.",
        "Bring me a unicorn from the kitchen.",
        "Bring me the bottle from Narnia.",
    ],
)
def test_explicit_unknown_catalog_entity_is_unsupported(nlu_engine, text):
    from src.dialogue import DialogueManager

    result = DialogueManager(engine=nlu_engine).process(text)
    assert result.status == DialogueStatus.UNSUPPORTED
    planner = MagicMock(wraps=TaskPlanner())
    pipeline = AIPipeline(engine=nlu_engine, deterministic_planner=planner)
    pipeline_result = pipeline.run_text(text)
    assert pipeline_result.plan is None
    planner.plan.assert_not_called()


def test_unresolved_deictic_destination_stays_ambiguous(nlu_engine):
    result = DialogueManager(engine=nlu_engine).process("Go there.")
    assert result.status == DialogueStatus.CLARIFICATION_REQUIRED


def test_dialogue_negation_is_ready_stop(nlu_engine):
    manager = DialogueManager(engine=nlu_engine)
    result = manager.process("Do not go any further.")
    assert result.status == DialogueStatus.READY
    assert result.task["intent"] == "STOP"
def test_saved_model_recovers_bare_deictic_navigation():
    """The configured artifact should classify 'Go there' as navigation."""
    from src.nlu.nlu import NLUEngine

    engine = NLUEngine.load()
    result = engine.parse("Go there.")

    assert result.intent == "NAVIGATE"
    assert result.entities.get("action") == "navigate"
    assert "destination" not in result.entities

    dialogue = DialogueManager(engine=engine).process("Go there.")
    assert dialogue.status == DialogueStatus.CLARIFICATION_REQUIRED
