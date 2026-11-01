"""Validation requirements are loaded from config, not hardcoded per module."""

from src.validation.schema import load_policy


def test_policy_covers_six_robot_intents():
    policy = load_policy()
    assert policy.robot_intents == [
        "NAVIGATE",
        "RETRIEVE",
        "DELIVER",
        "FOLLOW",
        "STOP",
        "RETURN",
    ]
    assert "UNSUPPORTED" not in policy.robot_intents


def test_retrieve_requires_object_source_and_destination():
    spec = load_policy().requirements["RETRIEVE"]
    assert spec.required == ["object", "source_location", "destination"]


def test_stop_requires_nothing():
    assert load_policy().requirements["STOP"].required == []


def test_deliver_requires_object_and_person_or_destination():
    spec = load_policy().requirements["DELIVER"]
    assert spec.required == ["object"]
    assert spec.require_any == ["person", "destination"]


def test_user_is_allowed_destination_and_is_a_person_id():
    policy = load_policy()
    assert policy.allow_user_as_destination is True
    assert policy.user_value == "user"
    assert "user" in policy.person_ids
    assert "kitchen" in policy.location_ids
    assert "bottle" in policy.object_ids
