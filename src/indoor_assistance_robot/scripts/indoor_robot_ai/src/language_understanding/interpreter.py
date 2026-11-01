"""Adapters from existing dialogue output to typed language understanding."""

from __future__ import annotations

from typing import Dict

from src.dialogue.types import DialogueResult, DialogueStatus
from src.language_understanding.types import (
    EntityInterpretation,
    EntityProvenance,
    LanguageUnderstandingResult,
)
from src.validation.types import ValidationStatus


def _entity_provenance(dialogue: DialogueResult) -> Dict[str, EntityInterpretation]:
    """Infer conservative provenance from current deterministic state."""
    entities = dict(dialogue.task.get("entities") or {})
    provenance: Dict[str, EntityInterpretation] = {}
    original = dialogue.original_text or str(dialogue.task.get("original_text") or "")
    for slot, value in entities.items():
        source = EntityProvenance.EXPLICIT_UTTERANCE
        # Merged follow-up results preserve the original request text, so slots
        # absent from that text may have arrived through dialogue history.
        if (
            slot != "action"
            and value
            and value.replace("_", " ") not in original.lower()
        ):
            source = EntityProvenance.DIALOGUE_HISTORY
        provenance[slot] = EntityInterpretation(
            slot=slot,
            value=value,
            provenance=source,
            evidence=original or None,
        )
    return provenance


def from_dialogue_result(
    dialogue: DialogueResult,
    *,
    model_backend: str = "tfidf_logreg_baseline",
) -> LanguageUnderstandingResult:
    """Build a structured interpretation without changing planner contracts."""
    validation = dialogue.validation
    task = dict(dialogue.task or {})
    intent = task.get("intent") if isinstance(task.get("intent"), str) else None
    entities = dict(task.get("entities") or {})
    missing = list(validation.missing_entities if validation else [])
    ambiguity = list(validation.ambiguous_entities if validation else [])
    validation_status = validation.status.value if validation else None
    unsupported = []
    if validation is not None and validation.status == ValidationStatus.UNSUPPORTED:
        unsupported = list(validation.reasons)

    proposed = None
    if dialogue.status == DialogueStatus.READY:
        proposed = {
            "intent": intent,
            "entities": dict(entities),
            "ready_for_planning": True,
        }

    interpreted = intent or "UNSUPPORTED"
    if entities:
        pairs = ", ".join(f"{slot}={value}" for slot, value in sorted(entities.items()))
        interpreted = f"{interpreted} ({pairs})"

    return LanguageUnderstandingResult(
        original_utterance=dialogue.original_text,
        interpreted_request=interpreted,
        task_category=intent,
        entities=entities,
        entity_provenance=_entity_provenance(dialogue),
        missing_information=missing,
        ambiguity=ambiguity,
        clarification_required=dialogue.status
        in {
            DialogueStatus.CLARIFICATION_REQUIRED,
            DialogueStatus.CLARIFICATION_FAILED,
        },
        clarification_question=dialogue.question,
        relevant_references=list(dialogue.pending_slots),
        proposed_symbolic_task=proposed,
        unsupported_capabilities=unsupported,
        validation_status=validation_status,
        dialogue_status=dialogue.status.value,
        model_backend=model_backend,
        metadata={
            "intent_confidence": task.get("intent_confidence"),
            "clarification_count": dialogue.clarification_count,
            "validation_reasons": list(validation.reasons if validation else []),
        },
    )
