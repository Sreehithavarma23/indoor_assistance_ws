"""Deterministic clarification questions from config/dialogue.yaml."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Sequence

from src.config_loader import load_dialogue
from src.nlu.types import NLUResult
from src.validation.types import ValidationResult


@lru_cache(maxsize=1)
def _dialogue_config() -> dict:
    return load_dialogue()


def pending_slots_from_validation(validation: ValidationResult) -> List[str]:
    """Ordered unique slots that still need clarification."""
    ordered: List[str] = []
    for slot in list(validation.missing_entities) + list(validation.ambiguous_entities):
        if slot and slot not in ordered:
            ordered.append(slot)
    return ordered


def _format_template(template: str, entities: Dict[str, str]) -> str:
    safe = {
        "object": entities.get("object", "object"),
        "person": entities.get("person", "person"),
        "destination": entities.get("destination", "destination"),
        "source_location": entities.get("source_location", "location"),
        "action": entities.get("action", "action"),
    }
    try:
        return template.format(**safe)
    except (KeyError, ValueError):
        return template


def generate_clarification(
    nlu: NLUResult,
    validation: ValidationResult,
    pending_slots: Optional[Sequence[str]] = None,
) -> str:
    """Return one clarification question for the first pending slot."""
    config = _dialogue_config()
    slots = list(pending_slots) if pending_slots is not None else pending_slots_from_validation(validation)
    if not slots:
        return str(config.get("ambiguous_fallback", "Could you clarify what you mean?"))

    slot = slots[0]
    templates = config.get("templates") or {}
    intent_templates = templates.get(nlu.intent) or {}
    default_templates = templates.get("default") or {}

    template = intent_templates.get(slot) or default_templates.get(slot)
    if not template:
        return str(config.get("ambiguous_fallback", "Could you clarify what you mean?"))
    return _format_template(str(template), nlu.entities)
