"""Explicit text and alias normalization for NLU.

Rules (also listed in config/nlu.yaml):
- lowercase
- hyphens become spaces
- punctuation becomes spaces
- only aliases declared in config are accepted
- longest alias match wins
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

from src.config_loader import load_locations, load_nlu, load_objects, load_persons

AliasEntry = Tuple[str, str, str]  # alias_normalized, value, entity_type


@lru_cache(maxsize=1)
def _nlu_config() -> dict:
    return load_nlu()


def normalize_text(text: str) -> str:
    """Apply the documented surface-normalization rules."""
    settings = _nlu_config().get("normalization", {})
    cleaned = text.strip()
    if settings.get("lowercase", True):
        cleaned = cleaned.lower()
    if settings.get("replace_hyphens", True):
        cleaned = cleaned.replace("-", " ")
    # Straighten curly apostrophes, then drop in-word apostrophes so
    # "don't" / "don’t" become "dont" instead of the tokens "don" + "t".
    cleaned = cleaned.replace("’", "'").replace("‘", "'")
    cleaned = re.sub(r"(?<=[a-z0-9])'(?=[a-z])", "", cleaned)
    if settings.get("strip_punctuation", True):
        cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _entries_from_catalog(
    items: Sequence[Dict[str, object]],
    entity_type: str,
) -> List[AliasEntry]:
    entries: List[AliasEntry] = []
    for item in items:
        value = str(item.get("id", "")).strip()
        if not value:
            continue
        aliases = item.get("aliases") or [value.replace("_", " ")]
        if not isinstance(aliases, list):
            continue
        for alias in aliases:
            normalized = normalize_text(str(alias))
            if normalized:
                entries.append((normalized, value, entity_type))
    return entries


@lru_cache(maxsize=1)
def catalog_aliases() -> tuple:
    """All configured object / location / person / action aliases."""
    nlu = _nlu_config()
    entries: List[AliasEntry] = []
    entries.extend(_entries_from_catalog(load_objects().get("objects", []), "object"))
    entries.extend(_entries_from_catalog(load_locations().get("locations", []), "location"))
    entries.extend(_entries_from_catalog(load_persons().get("persons", []), "person"))
    entries.extend(_entries_from_catalog(nlu.get("actions", []), "action"))
    # Longest alias first so "kitchen table" beats "kitchen" and "table".
    entries.sort(key=lambda item: len(item[0].split()), reverse=True)
    return tuple(entries)


def alias_lookup(entity_type: str) -> Dict[str, str]:
    """Map a normalized alias to its configured id for one entity type."""
    return {
        alias: value
        for alias, value, kind in catalog_aliases()
        if kind == entity_type
    }


def tokens(text: str) -> List[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def find_alias_spans(
    text: str,
    entries: Iterable[AliasEntry] | None = None,
) -> List[Tuple[int, int, str, str, str]]:
    """Return non-overlapping (start, end, alias, value, type) token spans."""
    word_list = tokens(text)
    used = [False] * len(word_list)
    found: List[Tuple[int, int, str, str, str]] = []
    catalog = list(entries) if entries is not None else catalog_aliases()

    for alias, value, entity_type in catalog:
        alias_tokens = alias.split()
        width = len(alias_tokens)
        if width == 0:
            continue
        for start in range(0, len(word_list) - width + 1):
            if any(used[start:start + width]):
                continue
            if word_list[start:start + width] == alias_tokens:
                found.append((start, start + width, alias, value, entity_type))
                for index in range(start, start + width):
                    used[index] = True
    found.sort(key=lambda item: item[0])
    return found
