"""Intent generalization and unified NLU output tests."""

import pytest

from src.nlu.exceptions import NLUError
from src.nlu.nlu import NLUEngine, parse, parse_transcript


def _intent(nlu_engine, text: str) -> str:
    return nlu_engine.parse(text).intent


def test_navigate_unseen_phrasings(nlu_engine):
    assert _intent(nlu_engine, "Would you mind escorting me toward the kitchen?") == "NAVIGATE"
    assert _intent(nlu_engine, "Kindly proceed into the living room.") == "NAVIGATE"
    assert _intent(nlu_engine, "Mind walking me over to the kitchen?") == "NAVIGATE"


def test_bare_deictic_navigation_is_not_stop(nlu_engine):
    result = nlu_engine.parse("Go there.")
    assert result.intent == "NAVIGATE"
    assert result.intent != "STOP"


def test_retrieve_unseen_phrasings(nlu_engine):
    assert _intent(nlu_engine, "Mind bringing me the bottle?") == "RETRIEVE"
    assert _intent(nlu_engine, "Would you mind fetching the water bottle from the kitchen?") == "RETRIEVE"
    assert _intent(nlu_engine, "Could you kindly fetch me that water bottle?") == "RETRIEVE"


def test_deliver_unseen_phrasings(nlu_engine):
    assert _intent(nlu_engine, "Would you kindly pass the bottle along to John?") == "DELIVER"
    assert _intent(nlu_engine, "I'd like you to deliver the medicine to John.") == "DELIVER"


def test_follow_unseen_phrasings(nlu_engine):
    assert _intent(nlu_engine, "Can you shadow me for a bit?") == "FOLLOW"
    assert _intent(nlu_engine, "I'd like you to accompany me.") == "FOLLOW"


def test_stop_unseen_phrasings(nlu_engine):
    assert _intent(nlu_engine, "I need you to stop immediately.") == "STOP"
    assert _intent(nlu_engine, "Please pause for a second.") == "STOP"
    assert _intent(nlu_engine, "Please stop immediately if you can.") == "STOP"


def test_return_unseen_phrasings(nlu_engine):
    assert _intent(nlu_engine, "Kindly return to your charger.") == "RETURN"
    assert _intent(nlu_engine, "Go ahead and return home now.") == "RETURN"


def test_unsupported_commands(nlu_engine):
    assert _intent(nlu_engine, "What's the capital of France?") == "UNSUPPORTED"
    assert _intent(nlu_engine, "Calculate fifteen times forty.") == "UNSUPPORTED"
    assert _intent(nlu_engine, "Book a taxi.") == "UNSUPPORTED"
    assert _intent(nlu_engine, "Could you fly me to the roof?") == "UNSUPPORTED"
    assert _intent(nlu_engine, "Do a backflip.") == "UNSUPPORTED"


def test_empty_input_is_unsupported(nlu_engine):
    result = nlu_engine.parse("   ")
    assert result.intent == "UNSUPPORTED"
    assert result.intent_confidence == 0.0
    assert result.entities == {}


def test_malformed_input_raises(nlu_engine):
    with pytest.raises(NLUError):
        nlu_engine.parse(None)  # type: ignore[arg-type]


def test_unknown_configured_backend_is_rejected():
    with pytest.raises(NLUError, match="Unknown intent backend"):
        NLUEngine.load(backend="not-a-backend")


def test_output_structure_and_confidence(nlu_engine):
    result = nlu_engine.parse("Can you take me to the kitchen?")
    assert result.original_text == "Can you take me to the kitchen?"
    assert result.intent == "NAVIGATE"
    assert 0.0 <= result.intent_confidence <= 1.0
    assert result.entities["destination"] == "kitchen"
    assert "NAVIGATE" in result.intent_probabilities
    assert result.mentions


def test_multiple_entities_in_one_sentence(nlu_engine):
    result = nlu_engine.parse("Please bring me the water bottle from the kitchen.")
    assert result.intent == "RETRIEVE"
    assert result.entities["object"] == "water_bottle"
    assert result.entities["source_location"] == "kitchen"
    assert result.entities["destination"] == "user"


def test_same_intent_different_wording(nlu_engine):
    first = nlu_engine.parse("Go to the kitchen.")
    second = nlu_engine.parse("I want to go to the kitchen.")
    assert first.intent == second.intent == "NAVIGATE"
    assert first.entities["destination"] == second.entities["destination"] == "kitchen"


def test_public_parse_helper(nlu_engine):
    result = parse("Follow me.", engine=nlu_engine)
    assert result.intent == "FOLLOW"
    assert result.entities["person"] == "user"


def test_parse_transcript_uses_text_attribute(nlu_engine):
    class FakeTranscript:
        text = "Stop."

    result = parse_transcript(FakeTranscript(), engine=nlu_engine)
    assert result.intent == "STOP"


def test_does_not_mark_valid_status(nlu_engine):
    result = nlu_engine.parse("Bring me the bottle.")
    payload = result.to_dict()
    assert "VALID" not in payload
    assert payload.get("status") is None
