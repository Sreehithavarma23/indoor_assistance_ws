"""Phase 4 validation: required slots, deixis, catalogs, confidence policy."""

import pytest

from src.nlu.types import NLUResult
from src.validation import validate
from src.validation.exceptions import ValidationError
from src.validation.types import ValidationStatus
from tests.validation.conftest import make_nlu, mention


def test_navigate_kitchen_is_valid(nlu_engine):
    result = validate(nlu_engine.parse("Take me to the kitchen."))
    assert result.intent == "NAVIGATE"
    assert result.entities.get("destination") == "kitchen"
    assert result.status == ValidationStatus.VALID
    assert result.missing_entities == []
    assert result.ambiguous_entities == []


def test_follow_me_is_valid(nlu_engine):
    result = validate(nlu_engine.parse("Follow me."))
    assert result.intent == "FOLLOW"
    assert result.entities.get("person") == "user"
    assert result.status == ValidationStatus.VALID


def test_stop_is_valid_without_entities(nlu_engine):
    result = validate(nlu_engine.parse("Stop."))
    assert result.intent == "STOP"
    assert result.status == ValidationStatus.VALID
    assert result.missing_entities == []


def test_return_charging_station_is_valid(nlu_engine):
    result = validate(nlu_engine.parse("Return to the charging station."))
    assert result.intent == "RETURN"
    assert result.entities.get("destination") == "charging_station"
    assert result.status == ValidationStatus.VALID


def test_retrieve_full_command_is_valid(nlu_engine):
    result = validate(nlu_engine.parse("Bring me the bottle from the kitchen."))
    assert result.intent == "RETRIEVE"
    assert result.entities.get("object") == "bottle"
    assert result.entities.get("source_location") == "kitchen"
    assert result.entities.get("destination") == "user"
    assert result.status == ValidationStatus.VALID


def test_retrieve_without_source_is_incomplete(nlu_engine):
    result = validate(nlu_engine.parse("Bring me the bottle."))
    assert result.intent == "RETRIEVE"
    assert result.entities.get("object") == "bottle"
    assert result.entities.get("destination") == "user"
    assert result.status == ValidationStatus.INCOMPLETE
    assert "source_location" in result.missing_entities
    assert any("source_location" in reason for reason in result.reasons)
    assert not any("Where" in reason for reason in result.reasons)


def test_go_there_is_not_valid(nlu_engine):
    result = validate(nlu_engine.parse("Go there."))
    assert result.status != ValidationStatus.VALID
    assert result.status in {ValidationStatus.AMBIGUOUS, ValidationStatus.INCOMPLETE}


def test_go_there_as_navigate_is_ambiguous_deixis():
    nlu = make_nlu("Go there.", "NAVIGATE", {})
    result = validate(nlu)
    assert result.status == ValidationStatus.AMBIGUOUS
    assert "destination" in result.ambiguous_entities


def test_weather_is_unsupported(nlu_engine):
    result = validate(nlu_engine.parse("What's the weather today?"))
    assert result.status == ValidationStatus.UNSUPPORTED
    assert result.intent == "UNSUPPORTED"


def test_deliver_to_john_is_valid(nlu_engine):
    result = validate(nlu_engine.parse("Give the bottle to John."))
    assert result.intent == "DELIVER"
    assert result.entities.get("object") == "bottle"
    assert result.entities.get("person") == "john"
    assert result.status == ValidationStatus.VALID


def test_unknown_object_is_unsupported():
    nlu = make_nlu(
        "Bring me the unicorn from the kitchen.",
        "RETRIEVE",
        {
            "object": "unicorn",
            "source_location": "kitchen",
            "destination": "user",
        },
    )
    result = validate(nlu)
    assert result.status == ValidationStatus.UNSUPPORTED
    assert any("unicorn" in reason for reason in result.reasons)


def test_unknown_location_is_unsupported():
    nlu = make_nlu(
        "Go to the rooftop.",
        "NAVIGATE",
        {"destination": "rooftop"},
    )
    result = validate(nlu)
    assert result.status == ValidationStatus.UNSUPPORTED
    assert any("rooftop" in reason for reason in result.reasons)


def test_user_is_an_allowed_destination():
    nlu = make_nlu(
        "Bring me the bottle from the kitchen.",
        "RETRIEVE",
        {
            "object": "bottle",
            "source_location": "kitchen",
            "destination": "user",
        },
    )
    assert validate(nlu).status == ValidationStatus.VALID


def test_deliver_requires_person_or_destination():
    nlu = make_nlu("Give the bottle.", "DELIVER", {"object": "bottle"})
    result = validate(nlu)
    assert result.status == ValidationStatus.INCOMPLETE
    assert "person" in result.missing_entities
    assert "destination" in result.missing_entities


def test_follow_missing_person_is_incomplete():
    nlu = make_nlu("Follow.", "FOLLOW", {})
    result = validate(nlu)
    assert result.status == ValidationStatus.INCOMPLETE
    assert result.missing_entities == ["person"]


def test_navigate_missing_destination_without_deixis_is_incomplete():
    nlu = make_nlu("Go.", "NAVIGATE", {})
    result = validate(nlu)
    assert result.status == ValidationStatus.INCOMPLETE
    assert result.missing_entities == ["destination"]


def test_competing_robot_intents_are_ambiguous():
    nlu = make_nlu(
        "Please head over.",
        "NAVIGATE",
        {"destination": "kitchen"},
        confidence=0.28,
        probabilities={"NAVIGATE": 0.28, "RETURN": 0.27, "UNSUPPORTED": 0.10},
    )
    result = validate(nlu)
    assert result.status == ValidationStatus.AMBIGUOUS
    assert result.ambiguous_entities == []
    assert any("competing_margin" in reason for reason in result.reasons)


def test_low_confidence_with_clear_winner_is_valid():
    nlu = make_nlu(
        "Follow me.",
        "FOLLOW",
        {"person": "user"},
        confidence=0.22,
        probabilities={"FOLLOW": 0.22, "RETRIEVE": 0.10, "UNSUPPORTED": 0.09},
    )
    result = validate(nlu)
    assert result.status == ValidationStatus.VALID


def test_conflicting_destinations_are_ambiguous():
    nlu = make_nlu(
        "Take me to the kitchen or the office.",
        "NAVIGATE",
        {"destination": "kitchen"},
        mentions=[
            mention("destination", "kitchen", "location"),
            mention("destination", "office", "location"),
        ],
    )
    result = validate(nlu)
    assert result.status == ValidationStatus.AMBIGUOUS
    assert "destination" in result.ambiguous_entities


def test_it_without_object_is_ambiguous():
    nlu = make_nlu(
        "Bring it from the kitchen.",
        "RETRIEVE",
        {"source_location": "kitchen", "destination": "user"},
    )
    result = validate(nlu)
    assert result.status == ValidationStatus.AMBIGUOUS
    assert "object" in result.ambiguous_entities


def test_empty_nlu_is_unsupported():
    nlu = make_nlu("", "UNSUPPORTED", {}, confidence=0.0)
    assert validate(nlu).status == ValidationStatus.UNSUPPORTED


def test_validate_rejects_non_nlu_result():
    with pytest.raises(ValidationError):
        validate("not an nlu result")  # type: ignore[arg-type]


def test_phase3_nlu_result_has_no_status(nlu_engine):
    nlu = nlu_engine.parse("Bring me the bottle.")
    assert not hasattr(nlu, "status") or getattr(nlu, "status", None) is None
    assert isinstance(nlu, NLUResult)
