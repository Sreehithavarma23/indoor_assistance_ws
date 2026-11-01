"""Load the hand-authored Phase 3 text dataset.

This is a development set, not a real-world speech benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from src.config_loader import PROJECT_ROOT
from src.nlu.exceptions import DatasetError
from src.nlu.types import NLUExample

DATA_DIR = PROJECT_ROOT / "data" / "nlu"
SUPPORTED_INTENTS = {
    "NAVIGATE",
    "RETRIEVE",
    "DELIVER",
    "FOLLOW",
    "STOP",
    "RETURN",
    "UNSUPPORTED",
}


def _parse_record(record: dict, split: str, index: int) -> NLUExample:
    if "text" not in record or "intent" not in record:
        raise DatasetError(f"{split} example {index} is missing text or intent.")
    text = str(record["text"]).strip()
    intent = str(record["intent"]).strip()
    if not text:
        raise DatasetError(f"{split} example {index} has empty text.")
    if intent not in SUPPORTED_INTENTS:
        raise DatasetError(f"{split} example {index} has unknown intent {intent!r}.")
    entities = record.get("entities") or {}
    if not isinstance(entities, dict):
        raise DatasetError(f"{split} example {index} entities must be a mapping.")
    cleaned = {str(key): str(value) for key, value in entities.items() if value}
    return NLUExample(text=text, intent=intent, entities=cleaned, split=split)


def load_jsonl(path: Path, split: str) -> List[NLUExample]:
    if not path.is_file():
        raise DatasetError(f"Dataset file not found: {path}")
    examples: List[NLUExample] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path} line {index} is not valid JSON.") from exc
            examples.append(_parse_record(record, split, index))
    if not examples:
        raise DatasetError(f"{path} contains no examples.")
    return examples


def load_split(split: str) -> List[NLUExample]:
    """Load data/nlu/{split}.jsonl."""
    if split not in {"train", "test"}:
        raise DatasetError(f"Unknown split: {split}")
    return load_jsonl(DATA_DIR / f"{split}.jsonl", split)


def load_train() -> List[NLUExample]:
    return load_split("train")


def load_test() -> List[NLUExample]:
    return load_split("test")


def texts_and_intents(examples: Sequence[NLUExample]) -> tuple[List[str], List[str]]:
    return [item.text for item in examples], [item.intent for item in examples]


def dataset_summary(examples: Iterable[NLUExample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for example in examples:
        counts[example.intent] = counts.get(example.intent, 0) + 1
    return counts


def assert_no_overlap(train: Sequence[NLUExample], test: Sequence[NLUExample]) -> None:
    """Fail if the same normalized sentence appears in both splits."""
    from src.nlu.normalization import normalize_text

    train_texts = {normalize_text(item.text) for item in train}
    overlap = [item.text for item in test if normalize_text(item.text) in train_texts]
    if overlap:
        preview = "; ".join(overlap[:5])
        raise DatasetError(f"Train/test leakage: {preview}")
