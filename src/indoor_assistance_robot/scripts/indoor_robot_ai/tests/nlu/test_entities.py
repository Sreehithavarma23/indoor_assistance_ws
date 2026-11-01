"""Entity extraction baseline tests. No invented values."""

from src.nlu.entity_extractor import extract_entities


def test_object_and_source_and_user_destination():
    slots, mentions = extract_entities(
        "Please bring me the water bottle from the kitchen.",
        intent="RETRIEVE",
    )
    assert slots["object"] == "water_bottle"
    assert slots["source_location"] == "kitchen"
    assert slots["destination"] == "user"
    kinds = {item.entity_type for item in mentions}
    assert "object" in kinds
    assert "location" in kinds


def test_red_cup_from_living_room():
    slots, _ = extract_entities(
        "Could you get the red cup from the living room?",
        intent="RETRIEVE",
    )
    assert slots["object"] == "red_cup"
    assert slots["source_location"] == "living_room"


def test_kitchen_table_is_not_split():
    slots, mentions = extract_entities(
        "Bring the water bottle from the kitchen table.",
        intent="RETRIEVE",
    )
    assert slots["source_location"] == "kitchen_table"
    values = [item.value for item in mentions if item.entity_type == "location"]
    assert "kitchen" not in values
    assert "table" not in values


def test_navigate_destination():
    slots, _ = extract_entities("Take me to the kitchen.", intent="NAVIGATE")
    assert slots["destination"] == "kitchen"
    assert slots.get("object") is None


def test_take_me_to_does_not_mark_user_as_destination():
    slots, _ = extract_entities("Take me to the kitchen.", intent="NAVIGATE")
    assert slots.get("destination") == "kitchen"
    assert slots.get("person") is None


def test_follow_me_is_person_user():
    slots, _ = extract_entities("Follow me.", intent="FOLLOW")
    assert slots["person"] == "user"


def test_deliver_person_and_object():
    slots, _ = extract_entities("Give the bottle to John.", intent="DELIVER")
    assert slots["object"] == "bottle"
    assert slots["person"] == "john"


def test_action_extracted_when_configured():
    slots, _ = extract_entities("Please stop.", intent="STOP")
    assert slots["action"] == "stop"


def test_does_not_invent_unknown_object():
    slots, mentions = extract_entities("Bring me the unicorn.", intent="RETRIEVE")
    assert "object" not in slots
    assert all(item.value != "unicorn" for item in mentions)


def test_empty_text_has_no_entities():
    slots, mentions = extract_entities("", intent="STOP")
    assert slots == {}
    assert mentions == []


def test_lounge_normalizes_to_living_room():
    slots, _ = extract_entities("Go to the lounge.", intent="NAVIGATE")
    assert slots["destination"] == "living_room"
