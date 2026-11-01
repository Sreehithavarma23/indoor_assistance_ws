"""Deterministic entity/slot extractor.

This is a transparent baseline: configured aliases + cue phrases.
It is not a neural NER model and must not invent unconfigured values.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from src.logging_utils import get_logger
from src.nlu.normalization import _nlu_config, find_alias_spans, normalize_text, tokens
from src.nlu.types import EntityMention

logger = get_logger("nlu.entities")

_DETERMINERS = {"the", "a", "an", "this", "that", "my", "your"}

# When several action aliases fire, prefer intent-aligned / more specific ones.
# Generic movement verbs such as "go" must not override retrieve/return.
_INTENT_ACTION = {
    "NAVIGATE": "navigate",
    "RETRIEVE": "retrieve",
    "DELIVER": "deliver",
    "FOLLOW": "follow",
    "STOP": "stop",
    "RETURN": "return",
}
_ACTION_RANK = {
    "return": 40,
    "retrieve": 30,
    "deliver": 30,
    "follow": 30,
    "stop": 30,
    "navigate": 10,
}


def _cue_before(word_list: Sequence[str], start: int, cues: Sequence[str]) -> bool:
    # YAML 1.1 can parse unquoted words such as "off" as booleans; ignore those.
    cursor = start
    while cursor > 0 and word_list[cursor - 1] in _DETERMINERS:
        cursor -= 1
    prefix = word_list[:cursor]
    string_cues = [cue for cue in cues if isinstance(cue, str)]
    for cue in sorted(string_cues, key=lambda item: len(item.split()), reverse=True):
        cue_tokens = normalize_text(cue).split()
        width = len(cue_tokens)
        if width and len(prefix) >= width and list(prefix[-width:]) == cue_tokens:
            return True
    return False


def _contains_phrase(normalized: str, phrase: str) -> bool:
    pattern = r"\b" + re.escape(normalize_text(phrase)) + r"\b"
    return re.search(pattern, normalized) is not None


def _user_destination(normalized: str, settings: dict) -> bool:
    """True only for explicit retrieve/deliver cues, not 'take me to <place>'."""
    if re.search(r"\bto\s+me\b", normalized):
        return True
    rider_guard = bool(settings.get("rider_after_cue", True))
    for cue in settings.get("destination_cues", []):
        cue_norm = normalize_text(cue)
        if not cue_norm:
            continue
        pattern = r"\b" + re.escape(cue_norm) + r"\b"
        if rider_guard:
            pattern += r"(?!\s+to\b)"
        if re.search(pattern, normalized):
            return True
    return False


def _user_person(normalized: str, settings: dict) -> bool:
    if re.search(r"\bto\s+me\b", normalized):
        return True
    for cue in settings.get("person_cues", []):
        if _contains_phrase(normalized, cue):
            return True
    return False


def _is_discourse_go(mention: EntityMention, word_list: Sequence[str]) -> bool:
    """True for discourse 'go ahead', which is not a navigate command."""
    if mention.entity_type != "action" or mention.value != "navigate":
        return False
    if normalize_text(mention.text) != "go":
        return False
    return 0 <= mention.end < len(word_list) and word_list[mention.end] == "ahead"


def _resolve_action(
    action_mentions: Sequence[EntityMention],
    intent: Optional[str],
) -> Optional[str]:
    """Choose one action value when multiple aliases match."""
    if not action_mentions:
        return None
    ranked: List[Tuple[str, int]] = [
        (mention.value, mention.start) for mention in action_mentions
    ]
    if intent:
        wanted = _INTENT_ACTION.get(intent)
        for value, _start in ranked:
            if value == wanted:
                return value
        # Deliver utterances often use retrieve verbs ("gets", "bring").
        if intent == "DELIVER":
            for value, _start in ranked:
                if value in {"retrieve", "deliver"}:
                    return value
    return max(
        ranked,
        key=lambda item: (_ACTION_RANK.get(item[0], 0), item[1]),
    )[0]


def extract_mentions(text: str) -> List[EntityMention]:
    """Find configured aliases and assign a role when a cue is present."""
    nlu = _nlu_config()
    word_list = tokens(text)
    normalized = normalize_text(text)
    source_cues = nlu.get("source_cues", [])
    destination_cues = nlu.get("destination_cues", [])
    user_settings = nlu.get("user_reference", {})
    mentions: List[EntityMention] = []

    for start, end, alias, value, entity_type in find_alias_spans(text):
        role: Optional[str] = None
        if entity_type == "object":
            role = "object"
        elif entity_type == "action":
            role = "action"
        elif entity_type == "person":
            role = "person"
        elif entity_type == "location":
            if _cue_before(word_list, start, source_cues):
                role = "source_location"
            elif _cue_before(word_list, start, destination_cues):
                role = "destination"
            else:
                # A catalog location following "or" is an alternative for
                # the same slot as the immediately preceding grounded
                # location ("to the kitchen or the office"). Keep both
                # mentions in that role so validation can reject the
                # unresolved choice instead of silently selecting the first.
                prior = next(
                    (
                        item
                        for item in reversed(mentions)
                        if item.entity_type == "location"
                        and item.role in {"source_location", "destination"}
                        and item.end <= start
                    ),
                    None,
                )
                if prior is not None:
                    between = word_list[prior.end:start]
                    if (
                        between
                        and between[0] == "or"
                        and all(token in _DETERMINERS for token in between[1:])
                    ):
                        role = prior.role

        mention = EntityMention(
            entity_type=entity_type,
            text=alias,
            value=value,
            role=role,
            start=start,
            end=end,
        )
        if _is_discourse_go(mention, word_list):
            continue
        mentions.append(mention)

    user_value = str(user_settings.get("value", "user"))
    if _user_destination(normalized, user_settings):
        mentions.append(
            EntityMention(
                entity_type="person",
                text="me",
                value=user_value,
                role="destination",
            )
        )
    elif _user_person(normalized, user_settings):
        mentions.append(
            EntityMention(
                entity_type="person",
                text="me",
                value=user_value,
                role="person",
            )
        )

    logger.debug("Extracted %d mention(s) from %r", len(mentions), text)
    return mentions


def mentions_to_slots(
    mentions: Sequence[EntityMention],
    intent: Optional[str] = None,
    text: Optional[str] = None,
) -> Dict[str, str]:
    """Collapse mentions into unique slots. Does not invent missing values.

    If a location has no cue, a documented fallback uses intent:
    NAVIGATE/RETURN → destination, RETRIEVE → source_location.
    The fallback never overwrites a cue-assigned slot.

    When several action mentions appear, intent-aligned and more specific
    actions outrank generic navigate verbs (e.g. "go grab" → retrieve).
    """
    slots: Dict[str, str] = {}

    def fill(role: str, value: str) -> None:
        if role and value and role not in slots:
            slots[role] = value

    unassigned_locations: List[str] = []
    action_mentions: List[EntityMention] = []
    for mention in mentions:
        if mention.entity_type == "location" and mention.role is None:
            unassigned_locations.append(mention.value)
            continue
        if mention.role == "action":
            action_mentions.append(mention)
            continue
        if mention.role:
            fill(mention.role, mention.value)

    action_value = _resolve_action(action_mentions, intent)
    if action_value:
        fill("action", action_value)

    if unassigned_locations:
        leftover = unassigned_locations[0]
        if intent in {"NAVIGATE", "RETURN"}:
            fill("destination", leftover)
        elif intent == "RETRIEVE":
            fill("source_location", leftover)
        elif intent == "DELIVER":
            fill("destination", leftover)
        # FOLLOW / STOP / UNSUPPORTED: leave unassigned; mentions still retain it.

    # FOLLOW with an explicit "me"/"myself" grounds person=user even when the
    # surface verb was not listed as a dedicated person cue.
    if intent == "FOLLOW" and "person" not in slots and text:
        user_value = str(_nlu_config().get("user_reference", {}).get("value", "user"))
        if re.search(r"\b(?:me|myself)\b", normalize_text(text)):
            fill("person", user_value)

    return slots


def extract_entities(text: str, intent: Optional[str] = None) -> tuple[Dict[str, str], List[EntityMention]]:
    mentions = extract_mentions(text)
    return mentions_to_slots(mentions, intent=intent, text=text), mentions
