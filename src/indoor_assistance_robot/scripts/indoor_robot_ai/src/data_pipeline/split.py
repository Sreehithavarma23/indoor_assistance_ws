"""Family-aware leakage-resistant splits for NLU v2."""

from __future__ import annotations

import json
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.config_loader import PROJECT_ROOT
from src.data_pipeline.schema import SPLIT_NAMES, DatasetRecord
from src.nlu.normalization import normalize_text

SPLITS_DIR = PROJECT_ROOT / "data" / "nlu_v2" / "splits"


def _stable_parity(value: str) -> int:
    """Return deterministic parity independent of PYTHONHASHSEED."""
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 2


def _assign_families(
    families: Sequence[str],
    rng: random.Random,
    ratios: Tuple[float, float, float, float, float] = (0.60, 0.12, 0.12, 0.08, 0.08),
) -> Dict[str, str]:
    """Map family_id → split name using deterministic shuffle."""
    ordered = list(families)
    rng.shuffle(ordered)
    n = len(ordered)
    cuts = []
    acc = 0.0
    for ratio in ratios[:-1]:
        acc += ratio
        cuts.append(int(round(acc * n)))
    buckets = [
        ordered[: cuts[0]],
        ordered[cuts[0] : cuts[1]],
        ordered[cuts[1] : cuts[2]],
        ordered[cuts[2] : cuts[3]],
        ordered[cuts[3] :],
    ]
    mapping: Dict[str, str] = {}
    for split_name, bucket in zip(SPLIT_NAMES, buckets):
        for fam in bucket:
            mapping[fam] = split_name
    return mapping


def _force_oos_heavy_into_test_oos(records: Sequence[DatasetRecord], mapping: Dict[str, str]) -> None:
    """Prefer putting unsupported-only families into test_oos when possible."""
    by_family: Dict[str, List[DatasetRecord]] = defaultdict(list)
    for rec in records:
        by_family[rec.family_id].append(rec)
    oos_families = [
        fam
        for fam, group in by_family.items()
        if all(r.intent == "UNSUPPORTED" for r in group)
        and any(
            t in {"aux_oos", "chat", "unrelated", "phys_impossible", "household"}
            for r in group
            for t in r.tags
        )
    ]
    for fam in oos_families:
        if mapping.get(fam) in {"train", "dev"}:
            # Keep some OOS in train for learning rejection; move half to test_oos.
            if _stable_parity(fam) == 0:
                mapping[fam] = "test_oos"


def split_records(
    records: Sequence[DatasetRecord],
    seed: int = 42,
) -> Dict[str, List[DatasetRecord]]:
    families = sorted({r.family_id for r in records if r.family_id})
    rng = random.Random(seed)
    mapping = _assign_families(families, rng)
    _force_oos_heavy_into_test_oos(records, mapping)

    # Paraphrase-tagged families preferred for test_paraphrase.
    para_families = sorted(
        {
            r.family_id
            for r in records
            if "paraphrase" in r.tags or r.source == "synthetic_paraphrase"
        }
    )
    for fam in para_families:
        if mapping.get(fam) == "test_indomain" and _stable_parity(fam) == 0:
            mapping[fam] = "test_paraphrase"

    splits: Dict[str, List[DatasetRecord]] = {name: [] for name in SPLIT_NAMES}
    for rec in records:
        split = mapping.get(rec.family_id, "train")
        clone = DatasetRecord(
            text=rec.text,
            intent=rec.intent,
            entities=dict(rec.entities),
            source=rec.source,
            family_id=rec.family_id,
            tags=list(rec.tags),
            task_order=rec.task_order,
            notes=rec.notes,
            split=split,
        )
        splits[split].append(clone)
    return splits


def check_leakage(splits: Dict[str, List[DatasetRecord]]) -> List[str]:
    """Return human-readable leakage findings (exact + family)."""
    issues: List[str] = []
    norms: Dict[str, Dict[str, str]] = {}
    for name, rows in splits.items():
        for rec in rows:
            key = normalize_text(rec.text)
            norms.setdefault(key, {})[name] = rec.family_id

    for key, present in norms.items():
        if len(present) > 1:
            issues.append(
                f"Exact normalized duplicate across splits {sorted(present)}: {key!r}"
            )

    # Family must not appear in both train and any frozen test.
    train_families = {r.family_id for r in splits.get("train", [])}
    for test_name in ("test_indomain", "test_paraphrase", "test_oos"):
        overlap = train_families & {r.family_id for r in splits.get(test_name, [])}
        if overlap:
            issues.append(
                f"Family overlap train∩{test_name}: {len(overlap)} families "
                f"(examples: {sorted(overlap)[:5]})"
            )
    return issues


def write_splits(splits: Dict[str, List[DatasetRecord]], directory: Path | None = None) -> Path:
    out_dir = directory or SPLITS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for rec in rows:
                handle.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    meta = {
        "splits": {name: len(rows) for name, rows in splits.items()},
        "intent_counts": {
            name: dict(Counter(r.intent for r in rows)) for name, rows in splits.items()
        },
        "source_counts": {
            name: dict(Counter(r.source for r in rows)) for name, rows in splits.items()
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out_dir


def load_split(name: str, directory: Path | None = None) -> List[DatasetRecord]:
    if name not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {name}")
    path = (directory or SPLITS_DIR) / f"{name}.jsonl"
    rows: List[DatasetRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(
                DatasetRecord(
                    text=row["text"],
                    intent=row["intent"],
                    entities=row.get("entities") or {},
                    source=row.get("source", ""),
                    family_id=row.get("family_id", ""),
                    tags=list(row.get("tags") or []),
                    task_order=row.get("task_order"),
                    notes=row.get("notes", ""),
                    split=row.get("split", name),
                )
            )
    return rows
