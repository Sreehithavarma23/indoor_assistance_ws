"""Typed records for the leakage-aware NLU dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SPLIT_NAMES = (
    "train",
    "dev",
    "test_indomain",
    "test_paraphrase",
    "test_oos",
)

VALID_SOURCES = {
    "legacy_hand_authored",
    "human_authored",
    "synthetic_template",
    "synthetic_paraphrase",
    "asr_noise_augmented",
    "adversarial_edge",
    "public_aux_oos",
}


@dataclass
class DatasetRecord:
    """One annotated utterance with provenance for leakage-free splits."""

    text: str
    intent: str
    entities: Dict[str, str] = field(default_factory=dict)
    source: str = "synthetic_template"
    family_id: str = ""
    tags: List[str] = field(default_factory=list)
    task_order: Optional[List[Dict[str, Any]]] = None
    notes: str = ""
    split: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if not payload.get("task_order"):
            payload.pop("task_order", None)
        if not payload.get("notes"):
            payload.pop("notes", None)
        if not payload.get("split"):
            payload.pop("split", None)
        return payload
