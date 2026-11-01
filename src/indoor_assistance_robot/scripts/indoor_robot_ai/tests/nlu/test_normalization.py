"""Normalization uses only explicit config rules."""

from src.nlu.normalization import alias_lookup, normalize_text


def test_lowercase_and_hyphen_and_punctuation():
    assert normalize_text("Water-Bottle!") == "water bottle"


def test_contraction_apostrophes_stay_one_token():
    assert normalize_text("Don't go!") == "dont go"
    assert normalize_text("Don’t go!") == "dont go"
    assert normalize_text("I'd like you to follow me.") == "id like you to follow me"


def test_configured_object_aliases():
    objects = alias_lookup("object")
    assert objects["water bottle"] == "water_bottle"
    assert objects["bottle"] == "bottle"
    assert "flask" not in objects


def test_configured_location_aliases():
    locations = alias_lookup("location")
    assert locations["living room"] == "living_room"
    assert locations["lounge"] == "living_room"
    assert locations["kitchen table"] == "kitchen_table"
    assert "rooftop" not in locations


def test_bottle_is_not_silently_water_bottle():
    objects = alias_lookup("object")
    assert objects["bottle"] != "water_bottle"
