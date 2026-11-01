"""Typed language-understanding adapter tests."""

from src.language_understanding import from_dialogue_result


def test_ready_dialogue_has_symbolic_task(pipeline):
    dialogue = pipeline.new_session().dialogue.process("Take me to the kitchen.")
    result = from_dialogue_result(dialogue)
    assert result.task_category == "NAVIGATE"
    assert result.entities["destination"] == "kitchen"
    assert result.proposed_symbolic_task == {
        "intent": "NAVIGATE",
        "entities": {"action": "navigate", "destination": "kitchen"},
        "ready_for_planning": True,
    }
    assert result.clarification_required is False


def test_clarification_records_missing_information(pipeline):
    dialogue = pipeline.new_session().dialogue.process("Bring me the bottle.")
    result = from_dialogue_result(dialogue)
    assert result.task_category == "RETRIEVE"
    assert "source_location" in result.missing_information
    assert result.clarification_required is True
    assert result.clarification_question == "Where should I get the bottle from?"


def test_pipeline_serializes_language_understanding(pipeline):
    result = pipeline.run_text("Go there.")
    payload = result.to_dict()
    lu = payload["language_understanding"]
    assert lu["task_category"] == "NAVIGATE"
    assert lu["clarification_required"] is True
    assert "destination" in lu["ambiguity"]
