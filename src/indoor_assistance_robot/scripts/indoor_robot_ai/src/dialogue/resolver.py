"""Resolve follow-up answers and detect new commands / STOP interrupts.

Rules (deterministic):

1. STOP (interrupt intent) while awaiting → CANCEL (handled by manager).
2. If the utterance fills a pending slot via catalog / cue / bare "me" → MERGE.
3. Else if the utterance is a VALID robot command with a different intent → REPLACE.
4. Else if VALID same intent but fills no pending slot and looks complete → REPLACE.
5. Else → cannot resolve as answer (manager may re-ask or fail).

Does not modify Phase 3 NLU or Phase 4 validation rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Set

from src.config_loader import load_dialogue, load_locations, load_objects, load_persons
from src.nlu.normalization import find_alias_spans, normalize_text, tokens
from src.nlu.types import NLUResult
from src.validation.schema import load_policy
from src.validation.types import ValidationResult, ValidationStatus


class ResolveAction(str, Enum):
    MERGE = "MERGE"
    REPLACE = "REPLACE"
    INTERRUPT = "INTERRUPT"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class ResolveDecision:
    action: ResolveAction
    slot_updates: Dict[str, str]
    reason: str


@lru_cache(maxsize=1)
def _dialogue_config() -> dict:
    return load_dialogue()


def _catalog_ids(kind: str) -> Set[str]:
    if kind == "object":
        return {str(item["id"]) for item in load_objects().get("objects") or [] if item.get("id")}
    if kind == "location":
        return {str(item["id"]) for item in load_locations().get("locations") or [] if item.get("id")}
    if kind == "person":
        return {str(item["id"]) for item in load_persons().get("persons") or [] if item.get("id")}
    return set()


def _strip_cues(text: str) -> str:
    cues = [normalize_text(c) for c in (_dialogue_config().get("answer_cues") or [])]
    word_list = tokens(text)
    while word_list and word_list[0] in cues:
        word_list = word_list[1:]
    return " ".join(word_list)


def _user_token_match(text: str) -> bool:
    normalized = normalize_text(text)
    user_tokens = [normalize_text(t) for t in (_dialogue_config().get("user_tokens") or [])]
    return normalized in user_tokens


def extract_value_for_slot(text: str, slot: str, nlu: Optional[NLUResult] = None) -> Optional[str]:
    """Pull a catalog value for a pending slot from a short answer.

    Phase 5 only: bare "me"/"myself" → user when slot is person or destination.
    """
    policy = load_policy()
    if slot in {"person", "destination"} and _user_token_match(text):
        return policy.user_value

    # Prefer NLU entity if it already filled the pending slot.
    if nlu is not None and nlu.entities.get(slot):
        return nlu.entities[slot]

    # Catalog mention mapped to the pending slot type.
    spans = find_alias_spans(text)
    if slot == "object":
        for _s, _e, _alias, value, kind in spans:
            if kind == "object":
                return value
    if slot in {"source_location", "destination"}:
        for _s, _e, _alias, value, kind in spans:
            if kind == "location":
                return value
        # destination may also be user from persons catalog in rare cases
        if slot == "destination":
            for _s, _e, _alias, value, kind in spans:
                if kind == "person" and value == policy.user_value:
                    return value
    if slot == "person":
        for _s, _e, _alias, value, kind in spans:
            if kind == "person":
                return value

    # NLU may have put a location under destination while we need source_location.
    if nlu is not None:
        if slot == "source_location":
            for key in ("source_location", "destination"):
                value = nlu.entities.get(key)
                if value and value in _catalog_ids("location"):
                    return value
        if slot == "destination":
            for key in ("destination", "source_location"):
                value = nlu.entities.get(key)
                if value and (
                    value in _catalog_ids("location")
                    or (policy.allow_user_as_destination and value == policy.user_value)
                ):
                    return value
        if slot == "person":
            value = nlu.entities.get("person")
            if value and value in _catalog_ids("person"):
                return value
        if slot == "object":
            value = nlu.entities.get("object")
            if value and value in _catalog_ids("object"):
                return value

    # Last resort: stripped text as a single catalog alias.
    stripped = _strip_cues(text)
    if not stripped:
        return None
    spans = find_alias_spans(stripped)
    if slot == "object":
        for _s, _e, _alias, value, kind in spans:
            if kind == "object":
                return value
    if slot in {"source_location", "destination"}:
        for _s, _e, _alias, value, kind in spans:
            if kind == "location":
                return value
    if slot == "person":
        for _s, _e, _alias, value, kind in spans:
            if kind == "person":
                return value
    return None


def fill_pending_slots(
    text: str,
    pending_slots: Sequence[str],
    nlu: Optional[NLUResult] = None,
) -> Dict[str, str]:
    """Fill as many pending slots as the answer supports (usually one)."""
    updates: Dict[str, str] = {}
    for slot in pending_slots:
        value = extract_value_for_slot(text, slot, nlu=nlu)
        if value:
            updates[slot] = value
            # One answer typically fills one slot; stop after first success
            # unless require_any alternatives share the same answer type.
            break
    # Special case: DELIVER pending person OR destination — one answer fills one.
    return updates


def merge_entities(
    base: Dict[str, str],
    updates: Dict[str, str],
) -> Dict[str, str]:
    """Preserve known entities; apply updates without inventing extras."""
    merged = dict(base)
    for key, value in updates.items():
        if value:
            merged[key] = value
    return merged


def build_merged_nlu(
    active: NLUResult,
    merged_entities: Dict[str, str],
    follow_up_text: str,
) -> NLUResult:
    """Synthetic NLUResult for re-validation after structured merge."""
    return NLUResult(
        original_text=active.original_text,
        intent=active.intent,
        intent_confidence=active.intent_confidence,
        entities=merged_entities,
        mentions=[],
        intent_probabilities=dict(active.intent_probabilities),
    )


def is_interrupt(nlu: NLUResult, validation: ValidationResult) -> bool:
    interrupts = {str(i) for i in (_dialogue_config().get("interrupt_intents") or ["STOP"])}
    return nlu.intent in interrupts and validation.status == ValidationStatus.VALID


def is_replacement_command(
    active: NLUResult,
    new_nlu: NLUResult,
    new_validation: ValidationResult,
    pending_slots: Sequence[str],
    slot_updates: Dict[str, str],
) -> bool:
    """True when the utterance should replace the pending task.

    A VALID command with a different robot intent replaces.
    A VALID command that did not fill any pending slot also replaces
    (e.g. restating a complete new command of the same family).
    """
    policy = load_policy()
    if new_nlu.intent not in policy.robot_intents:
        return False
    if new_validation.status != ValidationStatus.VALID:
        return False
    if slot_updates:
        # Successfully answered clarification — never replace.
        return False
    if new_nlu.intent != active.intent:
        return True
    # Same intent, VALID, and did not fill pending → treat as new full command.
    return True


def is_short_answer(text: str) -> bool:
    """True for fragments like 'the kitchen' / 'John' / 'me', not full commands."""
    stripped = _strip_cues(text)
    word_list = stripped.split() if stripped else []
    if len(word_list) <= 4:
        # Reject if an action verb plus a location/person pattern looks command-like.
        action_aliases = set()
        from src.config_loader import load_nlu

        for item in load_nlu().get("actions") or []:
            for alias in item.get("aliases") or []:
                action_aliases.add(normalize_text(str(alias)))
        # "take me to the bedroom" after cue strip still has take/me/to/bedroom
        if len(word_list) >= 3 and word_list[0] in action_aliases:
            return False
        return True
    return False


def decide_follow_up(
    text: str,
    active: NLUResult,
    pending_slots: Sequence[str],
    new_nlu: NLUResult,
    new_validation: ValidationResult,
) -> ResolveDecision:
    """Decide MERGE / REPLACE / INTERRUPT / UNRESOLVED for a follow-up turn."""
    if is_interrupt(new_nlu, new_validation):
        return ResolveDecision(ResolveAction.INTERRUPT, {}, "User issued STOP.")

    updates = fill_pending_slots(text, pending_slots, nlu=new_nlu)
    policy = load_policy()
    different_valid = (
        new_validation.status == ValidationStatus.VALID
        and new_nlu.intent in policy.robot_intents
        and new_nlu.intent != active.intent
    )

    # Full new command with a different VALID intent replaces, even if the
    # sentence contains a location that could fill the pending slot
    # (e.g. "take me to the bedroom" must not become source_location=bedroom).
    if different_valid and not is_short_answer(text):
        return ResolveDecision(
            ResolveAction.REPLACE,
            {},
            f"New VALID {new_nlu.intent} command replaces pending {active.intent}.",
        )

    if updates:
        return ResolveDecision(
            ResolveAction.MERGE,
            updates,
            f"Filled pending slot(s): {sorted(updates)}.",
        )

    if is_replacement_command(active, new_nlu, new_validation, pending_slots, updates):
        return ResolveDecision(
            ResolveAction.REPLACE,
            {},
            f"New VALID {new_nlu.intent} command replaces pending {active.intent}.",
        )

    return ResolveDecision(
        ResolveAction.UNRESOLVED,
        {},
        "Could not map utterance to pending slot or a replacement command.",
    )
