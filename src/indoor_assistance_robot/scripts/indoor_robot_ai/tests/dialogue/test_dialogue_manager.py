"""Phase 5 dialogue manager: clarify, merge, interrupt, replace."""

from src.dialogue import DialogueManager, DialogueStatus
from src.dialogue.clarifier import generate_clarification
from src.nlu.types import NLUResult
from src.validation import validate


def test_navigate_is_ready(manager):
    result = manager.process("Take me to the kitchen.")
    assert result.status == DialogueStatus.READY
    assert result.task["intent"] == "NAVIGATE"
    assert result.task["entities"]["destination"] == "kitchen"
    assert result.question is None


def test_retrieve_asks_for_source(manager):
    result = manager.process("Bring me the bottle.")
    assert result.status == DialogueStatus.CLARIFICATION_REQUIRED
    assert "source_location" in result.pending_slots
    assert result.question is not None
    assert "bottle" in result.question.lower()
    assert "where" in result.question.lower()


def test_retrieve_follow_up_from_kitchen(manager):
    manager.process("Bring me the bottle.")
    result = manager.process("From the kitchen.")
    assert result.status == DialogueStatus.READY
    assert result.task["intent"] == "RETRIEVE"
    assert result.task["entities"]["object"] == "bottle"
    assert result.task["entities"]["source_location"] == "kitchen"
    assert result.task["entities"]["destination"] == "user"


def test_the_kitchen_merges_as_source_not_navigate(manager):
    manager.process("Bring me the bottle.")
    result = manager.process("The kitchen.")
    assert result.status == DialogueStatus.READY
    assert result.task["intent"] == "RETRIEVE"
    assert result.task["entities"]["source_location"] == "kitchen"
    assert result.task["intent"] != "NAVIGATE"


def test_deliver_john_ready(manager):
    result = manager.process("Give the bottle to John.")
    assert result.status == DialogueStatus.READY
    assert result.task["entities"]["person"] == "john"


def test_deliver_someone_then_john(manager):
    first = manager.process("Give the bottle to someone.")
    assert first.status == DialogueStatus.CLARIFICATION_REQUIRED
    result = manager.process("John.")
    assert result.status == DialogueStatus.READY
    assert result.task["intent"] == "DELIVER"
    assert result.task["entities"]["object"] == "bottle"
    assert result.task["entities"]["person"] == "john"


def test_follow_me_ready(manager):
    result = manager.process("Follow me.")
    assert result.status == DialogueStatus.READY
    assert result.task["entities"]["person"] == "user"


def test_follow_pending_then_me(manager):
    first = manager.process("Follow.")
    assert first.status == DialogueStatus.CLARIFICATION_REQUIRED
    assert "person" in first.pending_slots
    result = manager.process("Me.")
    assert result.status == DialogueStatus.READY
    assert result.task["intent"] == "FOLLOW"
    assert result.task["entities"]["person"] == "user"


def test_unsupported_weather(manager):
    result = manager.process("What's the weather today?")
    assert result.status == DialogueStatus.UNSUPPORTED
    assert result.question is None


def test_stop_cancels_pending(manager):
    manager.process("Bring me the bottle.")
    result = manager.process("Stop.")
    assert result.status == DialogueStatus.CANCELLED
    assert result.task["intent"] == "STOP"


def test_new_navigate_replaces_retrieve(manager):
    manager.process("Bring me the bottle.")
    result = manager.process("Actually, take me to the bedroom.")
    assert result.status == DialogueStatus.READY
    assert result.task["intent"] == "NAVIGATE"
    assert result.task["entities"]["destination"] == "bedroom"
    assert "bottle" not in result.task["entities"]


def test_go_there_not_ready_with_invented_destination(manager):
    result = manager.process("Go there.")
    assert result.status != DialogueStatus.READY
    destination = result.task.get("entities", {}).get("destination")
    assert destination not in {"there", "here"}


def test_clarification_failed_after_max_turns(manager, monkeypatch):
    from src.dialogue import dialogue_manager as dm

    monkeypatch.setattr(
        dm,
        "_dialogue_config",
        lambda: {
            "max_clarification_turns": 1,
            "history_size": 8,
            "interrupt_intents": ["STOP"],
            "templates": {},
            "ambiguous_fallback": "Please clarify.",
        },
    )
    # Bypass lru_cache on the real loader used by clarifier too
    first = manager.process("Bring me the bottle.")
    assert first.status == DialogueStatus.CLARIFICATION_REQUIRED
    # Unresolvable answer should re-ask / fail under max=1
    result = manager.process("asdfgh qwerty")
    assert result.status in {
        DialogueStatus.CLARIFICATION_FAILED,
        DialogueStatus.CLARIFICATION_REQUIRED,
    }
    if result.status == DialogueStatus.CLARIFICATION_REQUIRED:
        result = manager.process("asdfgh qwerty")
    assert result.status == DialogueStatus.CLARIFICATION_FAILED


def test_multi_slot_retrieve_step_by_step(manager):
    first = manager.process("Bring me something.")
    assert first.status == DialogueStatus.CLARIFICATION_REQUIRED
    second = manager.process("The bottle.")
    # May still need source
    if second.status == DialogueStatus.CLARIFICATION_REQUIRED:
        third = manager.process("From the kitchen.")
        assert third.status == DialogueStatus.READY
        assert third.task["entities"]["object"] == "bottle"
        assert third.task["entities"]["source_location"] == "kitchen"
        assert third.task["entities"]["destination"] == "user"
    else:
        assert second.status == DialogueStatus.READY


def test_clarifier_uses_object_in_template():
    nlu = NLUResult(
        original_text="Bring me the bottle.",
        intent="RETRIEVE",
        intent_confidence=0.9,
        entities={"object": "bottle", "destination": "user"},
    )
    validation = validate(nlu)
    question = generate_clarification(nlu, validation)
    assert "bottle" in question.lower()


def test_reset_clears_state(manager):
    manager.process("Bring me the bottle.")
    manager.reset()
    assert manager.state.awaiting_clarification is False
    result = manager.process("Take me to the kitchen.")
    assert result.status == DialogueStatus.READY
