"""Action and person entity extraction regressions."""

from __future__ import annotations

from src.nlu.entity_extractor import extract_entities


def test_escorting_maps_to_navigate_action():
    slots, _ = extract_entities(
        "Would you mind escorting me toward the kitchen?",
        intent="NAVIGATE",
    )
    assert slots.get("action") == "navigate"
    assert slots.get("destination") == "kitchen"


def test_fetching_maps_to_retrieve_action():
    slots, _ = extract_entities(
        "Would you mind fetching the water bottle from the kitchen?",
        intent="RETRIEVE",
    )
    assert slots.get("action") == "retrieve"
    assert slots.get("object") == "water_bottle"
    assert slots.get("source_location") == "kitchen"


def test_go_grab_prefers_retrieve_over_navigate():
    slots, _ = extract_entities(
        "Go grab the medicine from the bedroom.",
        intent="RETRIEVE",
    )
    assert slots.get("action") == "retrieve"
    assert slots.get("object") == "medicine"


def test_bringing_me_extracts_retrieve_and_user():
    slots, _ = extract_entities(
        "Mind bringing me the bottle?",
        intent="RETRIEVE",
    )
    assert slots.get("action") == "retrieve"
    assert slots.get("object") == "bottle"
    assert slots.get("destination") == "user"


def test_gets_as_retrieve_for_deliver():
    slots, _ = extract_entities(
        "Please see that the nurse gets the water bottle.",
        intent="DELIVER",
    )
    assert slots.get("action") == "retrieve"
    assert slots.get("object") == "water_bottle"
    assert slots.get("person") == "nurse"


def test_heading_back_extracts_navigate_action():
    slots, _ = extract_entities(
        "Would you mind heading back to the charging station?",
        intent="RETURN",
    )
    # Surface verb is heading→navigate; return intent may still win slot via
    # specificity when "return" is absent. Gold labels navigate here.
    assert slots.get("action") == "navigate"
    assert slots.get("destination") == "charging_station"


def test_go_ahead_return_prefers_return_action():
    slots, _ = extract_entities(
        "Go ahead and return home now.",
        intent="RETURN",
    )
    assert slots.get("action") == "return"
    assert slots.get("destination") == "home"


def test_shadow_me_grounds_person_user():
    slots, _ = extract_entities(
        "Can you shadow me for a bit?",
        intent="FOLLOW",
    )
    assert slots.get("action") == "follow"
    assert slots.get("person") == "user"


def test_accompany_me_grounds_person_user():
    slots, _ = extract_entities(
        "I'd like you to accompany me.",
        intent="FOLLOW",
    )
    assert slots.get("action") == "follow"
    assert slots.get("person") == "user"


def test_go_there_still_has_navigate_action():
    slots, _ = extract_entities("Go there.", intent="NAVIGATE")
    assert slots.get("action") == "navigate"
