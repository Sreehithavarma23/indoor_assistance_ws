"""Resolver unit tests with constructed pending context."""

from src.dialogue.resolver import (
    ResolveAction,
    decide_follow_up,
    extract_value_for_slot,
    fill_pending_slots,
)
from src.nlu.types import NLUResult
from src.validation import validate


def _nlu(text: str, intent: str, entities: dict, confidence: float = 0.8) -> NLUResult:
    return NLUResult(
        original_text=text,
        intent=intent,
        intent_confidence=confidence,
        entities=entities,
        intent_probabilities={intent: confidence},
    )


def test_extract_kitchen_for_source_location():
    assert extract_value_for_slot("From the kitchen.", "source_location") == "kitchen"
    assert extract_value_for_slot("The kitchen.", "source_location") == "kitchen"


def test_extract_me_for_person():
    assert extract_value_for_slot("Me.", "person") == "user"
    assert extract_value_for_slot("myself", "person") == "user"


def test_extract_john_for_person():
    assert extract_value_for_slot("John.", "person") == "john"


def test_fill_pending_source():
    updates = fill_pending_slots("From the kitchen.", ["source_location"])
    assert updates == {"source_location": "kitchen"}


def test_decide_merge_over_navigate_parse(nlu_engine):
    active = _nlu(
        "Bring me the bottle.",
        "RETRIEVE",
        {"object": "bottle", "destination": "user"},
    )
    new_nlu = nlu_engine.parse("The kitchen.")
    new_validation = validate(new_nlu)
    decision = decide_follow_up(
        "The kitchen.",
        active,
        ["source_location"],
        new_nlu,
        new_validation,
    )
    assert decision.action == ResolveAction.MERGE
    assert decision.slot_updates["source_location"] == "kitchen"


def test_decide_interrupt_stop(nlu_engine):
    active = _nlu(
        "Bring me the bottle.",
        "RETRIEVE",
        {"object": "bottle", "destination": "user"},
    )
    new_nlu = nlu_engine.parse("Stop.")
    new_validation = validate(new_nlu)
    decision = decide_follow_up(
        "Stop.",
        active,
        ["source_location"],
        new_nlu,
        new_validation,
    )
    assert decision.action == ResolveAction.INTERRUPT
