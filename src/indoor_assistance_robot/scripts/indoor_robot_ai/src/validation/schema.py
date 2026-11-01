"""Load required-slot and confidence policy from config/validation.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Set

from src.config_loader import (
    load_locations,
    load_nlu,
    load_objects,
    load_persons,
    load_validation,
)


@dataclass(frozen=True)
class IntentRequirements:
    required: List[str] = field(default_factory=list)
    require_any: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationPolicy:
    robot_intents: List[str]
    requirements: Dict[str, IntentRequirements]
    competing_margin: float
    allow_user_as_destination: bool
    user_value: str
    unresolved_references: Dict[str, List[str]]
    object_ids: Set[str]
    location_ids: Set[str]
    person_ids: Set[str]


def _catalog_ids(items: list) -> Set[str]:
    return {str(item.get("id")) for item in items if item.get("id")}


@lru_cache(maxsize=1)
def load_policy() -> ValidationPolicy:
    raw = load_validation()
    nlu = load_nlu()
    user_value = str(nlu.get("user_reference", {}).get("value", "user"))

    requirements: Dict[str, IntentRequirements] = {}
    for intent, spec in (raw.get("requirements") or {}).items():
        spec = spec or {}
        requirements[str(intent)] = IntentRequirements(
            required=[str(slot) for slot in spec.get("required") or []],
            require_any=[str(slot) for slot in spec.get("require_any") or []],
        )

    refs = {
        str(slot): [str(word) for word in words]
        for slot, words in (raw.get("unresolved_references") or {}).items()
    }
    confidence = raw.get("confidence") or {}
    return ValidationPolicy(
        robot_intents=[str(item) for item in raw.get("robot_intents") or []],
        requirements=requirements,
        competing_margin=float(confidence.get("competing_margin", 0.05)),
        allow_user_as_destination=bool(raw.get("allow_user_as_destination", True)),
        user_value=user_value,
        unresolved_references=refs,
        object_ids=_catalog_ids(load_objects().get("objects") or []),
        location_ids=_catalog_ids(load_locations().get("locations") or []),
        person_ids=_catalog_ids(load_persons().get("persons") or []),
    )
