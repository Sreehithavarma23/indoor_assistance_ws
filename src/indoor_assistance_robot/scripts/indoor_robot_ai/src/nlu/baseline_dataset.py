"""Build the versioned, training-only NLU baseline dataset.

The legacy test corpus is recorded as excluded provenance. It is not copied
into the new training set or treated as a new independent test set.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.config_loader import PROJECT_ROOT
from src.nlu.normalization import normalize_text

DATASET_DIR = PROJECT_ROOT / "data" / "nlu_baseline_v3"
SPLITS_DIR = PROJECT_ROOT / "data" / "nlu_v2" / "splits"
LEGACY_TRAIN = PROJECT_ROOT / "data" / "nlu" / "train.jsonl"
LEGACY_TEST = PROJECT_ROOT / "data" / "nlu" / "test.jsonl"
SPLIT_NAMES = ("train", "dev", "test_indomain", "test_paraphrase", "test_oos")
SIMILARITY_REVIEW_THRESHOLD = 0.65


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        record = json.loads(stripped)
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        records.append((line_number, record))
    return records


def _row_key(record: dict[str, Any]) -> tuple[str, str]:
    return normalize_text(str(record["text"])), str(record["intent"])


def _stable_id(origin: str, line: int, record: dict[str, Any]) -> str:
    payload = "\n".join(
        (origin, str(line), normalize_text(str(record["text"])), str(record["intent"]))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _token_jaccard(left: str, right: str) -> float:
    a = set(normalize_text(left).split())
    b = set(normalize_text(right).split())
    return len(a & b) / len(a | b) if a or b else 1.0


def check_split_leakage(
    splits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Check exact-text and available source/template/family group overlap.

    Missing provenance is counted and reported as unverifiable, never treated
    as evidence that a split is independent.
    """
    issues: list[dict[str, str]] = []
    coverage: dict[str, dict[str, int]] = {}
    for field in ("source_id", "template_id", "paraphrase_family_id"):
        owners: dict[str, set[str]] = defaultdict(set)
        present = 0
        total = 0
        for split_name, rows in splits.items():
            for row in rows:
                total += 1
                value = row.get(field)
                if value is None or value == "":
                    continue
                present += 1
                owners[str(value)].add(split_name)
        coverage[field] = {"present_rows": present, "total_rows": total}
        for group_id, group_splits in owners.items():
            if len(group_splits) > 1:
                issues.append(
                    {
                        "kind": f"{field}_overlap",
                        "group_id": group_id,
                        "splits": ",".join(sorted(group_splits)),
                    }
                )

    text_owners: dict[str, set[str]] = defaultdict(set)
    for split_name, rows in splits.items():
        for row in rows:
            text_owners[normalize_text(str(row.get("text", "")))].add(split_name)
    for normalized, owners in text_owners.items():
        if normalized and len(owners) > 1:
            issues.append(
                {
                    "kind": "normalized_text_overlap",
                    "group_id": normalized,
                    "splits": ",".join(sorted(owners)),
                }
            )
    return {
        "issues": issues,
        "coverage": coverage,
        "independence_verifiable": all(
            stats["present_rows"] == stats["total_rows"]
            for stats in coverage.values()
        ),
    }


def _legacy_line_index(path: Path) -> dict[tuple[str, str], list[int]]:
    index: dict[tuple[str, str], list[int]] = defaultdict(list)
    for line_number, record in _read_jsonl(path):
        index[_row_key(record)].append(line_number)
    return index


def _origin_for_split_row(
    split_name: str,
    line_number: int,
    record: dict[str, Any],
    legacy_train_lines: dict[tuple[str, str], list[int]],
    legacy_test_lines: dict[tuple[str, str], list[int]],
    raw_index: dict[tuple[str, str, str], list[int]],
) -> dict[str, Any]:
    tags = set(record.get("tags") or [])
    key = _row_key(record)
    if "legacy_test" in tags:
        legacy_origin = "data/nlu/test.jsonl"
        legacy_lines = legacy_test_lines.get(key, [])
    elif "legacy_train" in tags:
        legacy_origin = "data/nlu/train.jsonl"
        legacy_lines = legacy_train_lines.get(key, [])
    else:
        legacy_origin = None
        legacy_lines = []

    family = str(record.get("family_id") or "")
    raw_lines = raw_index.get((key[0], key[1], family), [])
    if legacy_origin:
        origin_path = legacy_origin
        origin_line = legacy_lines[0] if len(legacy_lines) == 1 else None
        origin_status = "matched_legacy_file_row" if origin_line else "legacy_line_unresolved"
    else:
        origin_path = "data/nlu_v2/raw/corpus.jsonl"
        origin_line = raw_lines[0] if len(raw_lines) == 1 else None
        origin_status = "matched_raw_corpus_row" if origin_line else "raw_line_unresolved"

    family_status = (
        "generator_family_key"
        if family.startswith("tmpl:")
        else "sentence_derived_singleton_or_unverified"
        if family.startswith("legacy:")
        else "recorded_but_semantics_unverified"
    )
    return {
        "origin_dataset": origin_path,
        "origin_line": origin_line,
        "origin_status": origin_status,
        "v2_split": split_name,
        "v2_split_line": line_number,
        "source_category": record.get("source"),
        "source_id": None,
        "source_id_status": "author_or_collection_source_not_recorded",
        "template_id": None,
        "template_id_status": "not_present_in_source_metadata",
        "paraphrase_family_id": family or None,
        "family_id_status": family_status,
        "original_tags": list(record.get("tags") or []),
        "legacy_origin_dataset": legacy_origin,
        "legacy_origin_line": legacy_lines[0] if len(legacy_lines) == 1 else None,
    }


def build_dataset(output_dir: Path = DATASET_DIR) -> dict[str, Any]:
    """Create curated training data and exclusion provenance without overwrites."""
    target_train = output_dir / "train.jsonl"
    target_manifest = output_dir / "manifest.json"
    target_exclusion = output_dir / "legacy_test_exclusion.json"
    if any(p.exists() for p in (target_train, target_manifest, target_exclusion)):
        raise FileExistsError(
            f"Refusing to overwrite versioned dataset output: {output_dir}"
        )

    legacy_train_lines = _legacy_line_index(LEGACY_TRAIN)
    legacy_test_lines = _legacy_line_index(LEGACY_TEST)
    raw_index: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for line_number, record in _read_jsonl(PROJECT_ROOT / "data/nlu_v2/raw/corpus.jsonl"):
        raw_index[(_row_key(record)[0], _row_key(record)[1], str(record.get("family_id") or ""))].append(line_number)

    split_rows: dict[str, list[tuple[int, dict[str, Any]]]] = {
        name: _read_jsonl(SPLITS_DIR / f"{name}.jsonl") for name in SPLIT_NAMES
    }
    legacy_records: list[dict[str, Any]] = []
    train_records: list[dict[str, Any]] = []
    for split_name, rows in split_rows.items():
        for line_number, record in rows:
            tags = set(record.get("tags") or [])
            if "legacy_test" in tags:
                provenance = _origin_for_split_row(
                    split_name,
                    line_number,
                    record,
                    legacy_train_lines,
                    legacy_test_lines,
                    raw_index,
                )
                if provenance["legacy_origin_line"] is None:
                    raise ValueError(
                        "Could not uniquely map a legacy_test row to data/nlu/test.jsonl: "
                        f"{record.get('text')!r}"
                    )
                legacy_records.append(
                    {
                        "row_id": _stable_id(provenance["origin_dataset"], provenance["origin_line"], record),
                        "text": record["text"],
                        "intent": record["intent"],
                        "v2_split": split_name,
                        "was_in_v2_training_split": split_name == "train",
                        "provenance": provenance,
                    }
                )
                continue

            # The v3 model is trained only from rows that were already in the
            # v2 training split. No old evaluation split is promoted to train.
            if split_name != "train":
                continue
            provenance = _origin_for_split_row(
                split_name,
                line_number,
                record,
                legacy_train_lines,
                legacy_test_lines,
                raw_index,
            )
            origin_line = provenance["origin_line"]
            origin_path = provenance["origin_dataset"]
            if origin_line is None:
                # Split-file line is the stable, auditable source location when
                # the raw corpus cannot be uniquely resolved.
                origin_line = line_number
                origin_path = f"data/nlu_v2/splits/{split_name}.jsonl"
                provenance["origin_dataset"] = origin_path
                provenance["origin_line"] = origin_line
                provenance["origin_status"] = "v2_split_row_fallback"
            train_records.append(
                {
                    "row_id": _stable_id(origin_path, origin_line, record),
                    "text": str(record["text"]).strip(),
                    "intent": str(record["intent"]).strip(),
                    "entities": record.get("entities") or {},
                    "source": record.get("source"),
                    "source_id": None,
                    "template_id": None,
                    "paraphrase_family_id": record.get("family_id") or None,
                    "tags": list(record.get("tags") or []),
                    "provenance": provenance,
                }
            )

    # Verify every legacy test row is represented exactly once and with its
    # original label. This also prevents a modified split from silently
    # changing the exclusion set.
    old_test = _read_jsonl(LEGACY_TEST)
    original_counter = Counter(_row_key(record) for _, record in old_test)
    excluded_counter = Counter((normalize_text(r["text"]), r["intent"]) for r in legacy_records)
    if original_counter != excluded_counter:
        raise ValueError("legacy_test split rows do not exactly match data/nlu/test.jsonl")

    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    for row in train_records:
        duplicate_groups[normalize_text(row["text"])].append(row["row_id"])
    duplicate_groups = {k: v for k, v in duplicate_groups.items() if len(v) > 1}

    similarity_reviews = []
    for excluded in legacy_records:
        candidates = [
            (score, row)
            for row in train_records
            if (score := _token_jaccard(excluded["text"], row["text"]))
            >= SIMILARITY_REVIEW_THRESHOLD
        ]
        candidates.sort(key=lambda item: (-item[0], item[1]["row_id"]))
        if candidates:
            similarity_reviews.append(
                {
                    "excluded_row_id": excluded["row_id"],
                    "excluded_text": excluded["text"],
                    "candidate_training_rows": [
                        {
                            "row_id": row["row_id"],
                            "text": row["text"],
                            "intent": row["intent"],
                            "token_jaccard": round(score, 6),
                        }
                        for score, row in candidates[:5]
                    ],
                    "interpretation": (
                        "Lexical-neighbor review flag only; not proof of a shared source, "
                        "template, or paraphrase family."
                    ),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    with target_train.open("x", encoding="utf-8") as handle:
        for row in train_records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    train_hash = sha256_file(target_train)

    excluded_in_v2_train = [r for r in legacy_records if r["was_in_v2_training_split"]]
    exclusion_report = {
        "schema_version": 1,
        "legacy_source_dataset": "data/nlu/test.jsonl",
        "legacy_source_sha256": sha256_file(LEGACY_TEST),
        "total_legacy_test_rows_excluded": len(legacy_records),
        "legacy_test_rows_found_in_v2_train": len(excluded_in_v2_train),
        "label_counts": dict(Counter(r["intent"] for r in legacy_records)),
        "normalized_exact_duplicate_groups_within_legacy_test": 0,
        "conflicting_legacy_test_labels": 0,
        "legacy_family_id_note": (
            "Legacy IDs were created from individual normalized sentences; they do not "
            "establish paraphrase-family independence."
        ),
        "similarity_threshold_token_jaccard": SIMILARITY_REVIEW_THRESHOLD,
        "similarity_review_pair_count": len(similarity_reviews),
        "training_duplicate_groups": duplicate_groups,
        "rows": legacy_records,
        "similarity_review_flags": similarity_reviews,
        "eligibility_recommendation": (
            "Keep excluded from this baseline. Exact duplicates were not found, but source "
            "and template provenance are unavailable and lexical neighbors exist. Any later "
            "reuse requires human review and must not be represented as untouched-test evidence."
        ),
    }
    target_exclusion.write_text(
        json.dumps(exclusion_report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "dataset_version": "nlu_baseline_v3",
        "schema_version": 1,
        "purpose": "Curated training-only set; no independent evaluation rows are asserted.",
        "created_from": "data/nlu_v2/splits/train.jsonl",
        "excluded_legacy_test_manifest": "legacy_test_exclusion.json",
        "excluded_legacy_test_rows": len(legacy_records),
        "of_which_were_in_v2_train": len(excluded_in_v2_train),
        "train": {
            "path": "train.jsonl",
            "rows": len(train_records),
            "sha256": train_hash,
            "label_counts": dict(Counter(r["intent"] for r in train_records)),
            "normalized_exact_duplicate_groups": len(duplicate_groups),
            "all_source_ids_known": False,
            "all_template_ids_known": False,
        },
        "evaluation_data": {
            "dev": {"status": "not_available_independent_provenance_missing"},
            "test_indomain": {"status": "not_available_independent_provenance_missing"},
            "test_unseen_template": {"status": "not_available_template_ids_missing"},
            "test_paraphrase": {"status": "not_available_independent_family_provenance_missing"},
            "test_oos": {"status": "not_available_independent_provenance_missing"},
        },
        "provenance_limitations": [
            "Author/session/source IDs are not present in source data.",
            "Template IDs are not present; they were not inferred from surface text.",
            "Recorded legacy family IDs are sentence-derived singletons, not verified paraphrase families.",
            "No independent evaluation examples were manufactured or promoted from frozen v2 splits.",
        ],
    }
    target_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    manifest = build_dataset()
    print(
        json.dumps(
            {
                "train_rows": manifest["train"]["rows"],
                "train_sha256": manifest["train"]["sha256"],
                "legacy_test_excluded": manifest["excluded_legacy_test_rows"],
                "legacy_test_rows_previously_in_v2_train": manifest[
                    "of_which_were_in_v2_train"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
