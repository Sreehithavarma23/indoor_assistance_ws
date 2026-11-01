"""Catalog-backed lexical material for reproducible command generation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

from src.config_loader import load_locations, load_objects, load_persons


@dataclass(frozen=True)
class CatalogEntry:
    entity_id: str
    surface: str


@lru_cache(maxsize=1)
def object_entries() -> Tuple[CatalogEntry, ...]:
    items: List[CatalogEntry] = []
    for row in load_objects().get("objects", []):
        eid = str(row["id"])
        for alias in row.get("aliases", []):
            items.append(CatalogEntry(eid, str(alias)))
    return tuple(items)


@lru_cache(maxsize=1)
def location_entries() -> Tuple[CatalogEntry, ...]:
    items: List[CatalogEntry] = []
    for row in load_locations().get("locations", []):
        eid = str(row["id"])
        for alias in row.get("aliases", []):
            items.append(CatalogEntry(eid, str(alias)))
    return tuple(items)


@lru_cache(maxsize=1)
def person_entries() -> Tuple[CatalogEntry, ...]:
    items: List[CatalogEntry] = []
    for row in load_persons().get("persons", []):
        eid = str(row["id"])
        # Keep "user" out of open generation; use me/myself templates instead.
        if eid == "user":
            continue
        for alias in row.get("aliases", []):
            items.append(CatalogEntry(eid, str(alias)))
    return tuple(items)


def primary_surfaces() -> Dict[str, List[CatalogEntry]]:
    """One primary surface per id plus a few aliases for diversity."""
    by_kind: Dict[str, Dict[str, CatalogEntry]] = {
        "object": {},
        "location": {},
        "person": {},
    }
    for entry in object_entries():
        by_kind["object"].setdefault(entry.entity_id, entry)
    for entry in location_entries():
        by_kind["location"].setdefault(entry.entity_id, entry)
    for entry in person_entries():
        by_kind["person"].setdefault(entry.entity_id, entry)
    return {k: list(v.values()) for k, v in by_kind.items()}
