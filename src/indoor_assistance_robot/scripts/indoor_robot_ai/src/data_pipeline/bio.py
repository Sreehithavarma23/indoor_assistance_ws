"""Export BIO token labels from catalog-grounded entity spans.

Used for span evaluation and optional token-classification training.
Span offsets come from the deterministic alias matcher — not invented IDs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.config_loader import PROJECT_ROOT
from src.data_pipeline.schema import DatasetRecord
from src.nlu.entity_extractor import extract_entities
from src.nlu.normalization import normalize_text, tokens

BIO_DIR = PROJECT_ROOT / "data" / "nlu_v2" / "bio"

# Map entity slots to BIO type names.
SLOT_TO_BIO = {
    "object": "OBJECT",
    "source_location": "SOURCE",
    "destination": "DESTINATION",
    "person": "PERSON",
}


def tokens_and_bio(text: str, intent: str | None = None) -> Tuple[List[str], List[str]]:
    """Produce word tokens and BIO tags using the deterministic extractor spans."""
    word_list = tokens(text)
    labels = ["O"] * len(word_list)
    _, mentions = extract_entities(text, intent=intent)
    # Longer spans first so nested aliases do not overwrite.
    ordered = sorted(mentions, key=lambda m: (m.end - m.start), reverse=True)
    for mention in ordered:
        bio_type = SLOT_TO_BIO.get(mention.entity_type)
        if bio_type is None:
            continue
        if mention.start < 0 or mention.end > len(word_list) or mention.start >= mention.end:
            continue
        # Skip if already labeled (prefer longer earlier spans).
        if any(labels[i] != "O" for i in range(mention.start, mention.end)):
            continue
        labels[mention.start] = f"B-{bio_type}"
        for i in range(mention.start + 1, mention.end):
            labels[i] = f"I-{bio_type}"
    return word_list, labels


def export_bio_jsonl(records: Sequence[DatasetRecord], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            toks, labs = tokens_and_bio(rec.text, intent=rec.intent)
            handle.write(
                json.dumps(
                    {
                        "text": rec.text,
                        "intent": rec.intent,
                        "tokens": toks,
                        "bio": labs,
                        "entities": rec.entities,
                        "family_id": rec.family_id,
                        "source": rec.source,
                        "split": rec.split,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def span_f1_from_bio(
    gold_bio: Sequence[str],
    pred_bio: Sequence[str],
) -> Dict[str, float]:
    """Exact-span micro P/R/F1 from BIO sequences."""

    def spans(tags: Sequence[str]) -> set:
        found = set()
        i = 0
        while i < len(tags):
            tag = tags[i]
            if tag.startswith("B-"):
                typ = tag[2:]
                j = i + 1
                while j < len(tags) and tags[j] == f"I-{typ}":
                    j += 1
                found.add((i, j, typ))
                i = j
            else:
                i += 1
        return found

    g = spans(gold_bio)
    p = spans(pred_bio)
    tp = len(g & p)
    fp = len(p - g)
    fn = len(g - p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
