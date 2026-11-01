"""Shared linguistic cues used by NLU recovery and validation safety checks."""

from __future__ import annotations

import re

from src.nlu.normalization import _nlu_config, normalize_text

# Negated movement: do not treat as a positive navigate command.
_NEGATED_MOVEMENT = re.compile(
    r"\b(?:do\s+not|dont)\s+"
    r"(?:go|move|walk|proceed|drive|navigate|head|roll|escort)\b"
)
_STOP_MOVING = re.compile(
    r"\b(?:stop|halt|cease)\s+(?:going|moving|walking|proceeding|driving)\b"
)

# Contradictory motion: go/navigate … but stay/remain/don't move.
_CONTRADICTION = re.compile(
    r"\b(?:go|take\s+me|head|navigate|drive|move|walk)\b.{0,40}\b"
    r"but\s+(?:stay|remain|dont\s+move|do\s+not\s+move|stop|wait\s+here)\b"
)
_POSITIVE_MOVEMENT = re.compile(
    r"\b(?:go|move|walk|proceed|drive|navigate|head|roll|escort)\b"
)


def has_negated_movement(text: str) -> bool:
    """True when the utterance forbids further movement (STOP, not NAVIGATE)."""
    normalized = normalize_text(text)
    if not normalized:
        return False
    return bool(_NEGATED_MOVEMENT.search(normalized) or _STOP_MOVING.search(normalized))


def has_contradictory_instruction(text: str) -> bool:
    """True for instructions that both request and cancel motion."""
    normalized = normalize_text(text)
    if not normalized:
        return False
    if _CONTRADICTION.search(normalized):
        return True

    # Also catch an explicit positive movement command followed or preceded
    # by a negated movement command, e.g. "go to the kitchen and don't go
    # to the kitchen". The latter must not be collapsed into a plain STOP.
    if not _NEGATED_MOVEMENT.search(normalized):
        return False
    for match in _POSITIVE_MOVEMENT.finditer(normalized):
        prefix = normalized[: match.start()]
        if not re.search(r"\b(?:do\s+not|dont)\s+$", prefix):
            return True
    return False


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def matches_unsupported_capability(text: str) -> bool:
    """True when the text contains a configured out-of-skill phrase.

    Exact word-boundary match first. For single-token cues of length >= 5,
    also accept edit distance 1 (covers light ASR corruption such as baokflip).
    """
    normalized = normalize_text(text)
    if not normalized:
        return False
    words = normalized.split()
    phrases = _nlu_config().get("unsupported_capability_phrases") or []
    for phrase in phrases:
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        cue = normalize_text(phrase)
        if not cue:
            continue
        pattern = r"\b" + re.escape(cue) + r"\b"
        if re.search(pattern, normalized):
            return True
        cue_tokens = cue.split()
        if len(cue_tokens) == 1 and len(cue_tokens[0]) >= 5:
            target = cue_tokens[0]
            for word in words:
                if abs(len(word) - len(target)) <= 1 and _edit_distance(word, target) <= 1:
                    return True
    return False
