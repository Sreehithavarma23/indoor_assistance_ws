"""Individual validation checks. Each returns reasons / slot names only."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

from src.config_loader import load_locations, load_objects
from src.nlu.cues import (
    has_contradictory_instruction,
    has_negated_movement,
    matches_unsupported_capability,
)
from src.nlu.normalization import normalize_text
from src.nlu.types import EntityMention, NLUResult
from src.validation.schema import IntentRequirements, ValidationPolicy


def unsupported_capability_request(nlu: NLUResult) -> Tuple[bool, str]:
    """Safety net: configured out-of-skill phrases must not become robot tasks.

    Negated movement is STOP, not unsupported, so it is excluded here.
    """
    if has_negated_movement(nlu.original_text):
        return False, ""
    if matches_unsupported_capability(nlu.original_text):
        return True, (
            "Request matches an unsupported capability cue outside the "
            "robot's supported action space."
        )
    return False, ""


def competing_robot_intents(
    nlu: NLUResult,
    policy: ValidationPolicy,
) -> Tuple[bool, str]:
    """True when two robot intents are nearly tied.

    Low confidence alone is not treated as ambiguity: 7-way softmax on
    paraphrases is often diffuse even when argmax is stable.
    """
    if nlu.intent not in policy.robot_intents:
        return False, ""
    ranked = sorted(
        (
            (name, score)
            for name, score in nlu.intent_probabilities.items()
            if name in policy.robot_intents
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if len(ranked) < 2:
        return False, ""
    top_name, top_score = ranked[0]
    second_name, second_score = ranked[1]
    if top_name != nlu.intent:
        return False, ""
    margin = top_score - second_score
    if margin < policy.competing_margin:
        return True, (
            f"Robot intents {top_name} ({top_score:.3f}) and "
            f"{second_name} ({second_score:.3f}) differ by {margin:.3f}, "
            f"below competing_margin {policy.competing_margin:.3f}."
        )
    return False, ""


def missing_required_slots(
    entities: Dict[str, str],
    spec: IntentRequirements,
) -> Tuple[List[str], List[str]]:
    """Return (missing_slot_names, reason strings)."""
    missing: List[str] = []
    reasons: List[str] = []
    for slot in spec.required:
        if not entities.get(slot):
            missing.append(slot)
            reasons.append(f"Required entity '{slot}' is missing.")
    if spec.require_any and not any(entities.get(slot) for slot in spec.require_any):
        missing.extend(slot for slot in spec.require_any if slot not in missing)
        joined = " or ".join(spec.require_any)
        reasons.append(f"At least one of [{joined}] is required.")
    return missing, reasons


def _allowed_values(slot: str, policy: ValidationPolicy) -> set[str]:
    if slot == "object":
        return set(policy.object_ids)
    if slot in {"source_location"}:
        return set(policy.location_ids)
    if slot == "destination":
        allowed = set(policy.location_ids)
        if policy.allow_user_as_destination:
            allowed.add(policy.user_value)
        return allowed
    if slot == "person":
        return set(policy.person_ids)
    return set()


def unknown_catalog_values(
    entities: Dict[str, str],
    spec: IntentRequirements,
    policy: ValidationPolicy,
) -> List[str]:
    """Reasons for filled required/any slots whose values are not configured."""
    reasons: List[str] = []
    slots = list(spec.required) + list(spec.require_any)
    for slot in slots:
        value = entities.get(slot)
        if not value:
            continue
        allowed = _allowed_values(slot, policy)
        if allowed and value not in allowed:
            reasons.append(
                f"Entity '{slot}' value {value!r} is not in the configured catalog."
            )
    return reasons


def unknown_catalog_reference(nlu: NLUResult) -> str:
    """Detect an explicit object/place phrase that extraction could not ground.

    This is intentionally limited to clear slot cues. It does not guess an
    entity value or turn bare/deictic references into UNSUPPORTED.
    """
    text = normalize_text(nlu.original_text)
    if not text:
        return ""

    location_aliases = {
        normalize_text(alias)
        for item in load_locations().get("locations", [])
        for alias in item.get("aliases", [])
    }
    object_aliases = {
        normalize_text(alias)
        for item in load_objects().get("objects", [])
        for alias in item.get("aliases", [])
    }

    candidates: List[Tuple[str, re.Pattern[str], set[str]]] = []
    if nlu.intent in {"NAVIGATE", "RETURN"} and not nlu.entities.get("destination"):
        candidates.append((
            "destination",
            re.compile(r"\b(?:to|toward|towards|into)\s+(?:the|a|an)\s+([a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,2})"),
            location_aliases,
        ))
        # Also accept the common article-free form: "go to Narnia".
        if not re.search(r"\b(?:to|toward|towards|into)\s+(?:the|a|an)\b", text):
            candidates.append((
                "destination",
                re.compile(r"\b(?:to|toward|towards|into)\s+([a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,2})"),
                location_aliases,
            ))
    elif nlu.intent == "RETRIEVE":
        if not nlu.entities.get("object"):
            candidates.append((
                "object",
                re.compile(r"\b(?:bring|get|fetch|grab|retrieve|collect|obtain)\s+(?:me\s+)?(?:the|a|an)\s+([a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,2})"),
                object_aliases,
            ))
            if not re.search(r"\b(?:bring|get|fetch|grab|retrieve|collect|obtain)\s+(?:me\s+)?(?:the|a|an)\b", text):
                candidates.append((
                    "object",
                    re.compile(r"\b(?:bring|get|fetch|grab|retrieve|collect|obtain)\s+(?:me\s+)?([a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,2})"),
                    object_aliases,
                ))
        if not nlu.entities.get("source_location"):
            candidates.append((
                "source_location",
                re.compile(r"\b(?:from|at|in|inside)\s+(?:the|a|an)\s+([a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,2})"),
                location_aliases,
            ))
            if not re.search(r"\b(?:from|at|in|inside)\s+(?:the|a|an)\b", text):
                candidates.append((
                    "source_location",
                    re.compile(r"\b(?:from|at|in|inside)\s+([a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,2})"),
                    location_aliases,
                ))

    for slot, pattern, aliases in candidates:
        match = pattern.search(text)
        if not match:
            continue
        value = re.split(r"\b(?:and|but|then|from|at|in|for|to)\b", match.group(1))[0].strip()
        if not value or value in {
            "there", "here", "over there", "it", "that", "this", "one", "me",
            "something", "anything", "stuff", "thing",
        }:
            continue
        if any(value == alias or value.startswith(alias + " ") for alias in aliases):
            continue
        return f"Explicit {slot} phrase {value!r} is not in the configured catalog."
    return ""


def conflicting_mention_values(
    mentions: Sequence[EntityMention],
    spec: IntentRequirements,
) -> Tuple[List[str], List[str]]:
    """Two different grounded values for the same required role → ambiguous."""
    watched = set(spec.required) | set(spec.require_any)
    grouped: Dict[str, set[str]] = {}
    for mention in mentions:
        role = mention.role
        if role and role in watched and mention.value:
            grouped.setdefault(role, set()).add(mention.value)
    ambiguous: List[str] = []
    reasons: List[str] = []
    for role, values in grouped.items():
        if len(values) > 1:
            ambiguous.append(role)
            reasons.append(
                f"Multiple values for '{role}': {sorted(values)}."
            )
    return ambiguous, reasons


def unresolved_references(
    text: str,
    entities: Dict[str, str],
    spec: IntentRequirements,
    policy: ValidationPolicy,
) -> Tuple[List[str], List[str]]:
    """Required slot empty, but the text uses a deictic word for that slot."""
    normalized = normalize_text(text)
    if not normalized:
        return [], []
    watched = list(spec.required) + [
        slot for slot in spec.require_any if not any(entities.get(item) for item in spec.require_any)
    ]
    ambiguous: List[str] = []
    reasons: List[str] = []
    for slot in watched:
        if entities.get(slot):
            continue
        for phrase in policy.unresolved_references.get(slot, []):
            pattern = r"\b" + re.escape(normalize_text(phrase)) + r"\b"
            if re.search(pattern, normalized):
                if slot not in ambiguous:
                    ambiguous.append(slot)
                    reasons.append(
                        f"Unresolved reference {phrase!r} for '{slot}'."
                    )
                break
    return ambiguous, reasons
