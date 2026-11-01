"""Score the configured v3 artifact and record its exact input hashes.

The initial supported mode is explicitly a legacy diagnostic. It is not an
independent source/template evaluation split.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.config_loader import PROJECT_ROOT, load_nlu
from src.nlu.baseline_dataset import sha256_file
from src.nlu.exceptions import IntentModelError
from src.nlu.nlu import NLUEngine
from src.nlu.types import NLUResult
from src.validation import validate

LEGACY_TEST = PROJECT_ROOT / "data/nlu/test.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/nlu_baseline_v3/legacy_diagnostic.json"


def _load_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or "text" not in row or "intent" not in row:
            raise ValueError(f"Invalid labeled row at {path}:{line_number}")
        rows.append((line_number, row))
    return rows


def _metrics(gold: list[str], predicted: list[str], labels: list[str]) -> dict[str, Any]:
    confusion = {g: {p: 0 for p in labels} for g in labels}
    for actual, guess in zip(gold, predicted):
        confusion[actual][guess] += 1
    per_class = {}
    f1s = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(actual == label for actual in gold),
        }
    correct = sum(actual == guess for actual, guess in zip(gold, predicted))
    oos_tp = confusion["UNSUPPORTED"]["UNSUPPORTED"]
    oos_fp = sum(confusion[label]["UNSUPPORTED"] for label in labels if label != "UNSUPPORTED")
    oos_fn = sum(confusion["UNSUPPORTED"][label] for label in labels if label != "UNSUPPORTED")
    return {
        "accuracy": correct / len(gold) if gold else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "rows_gold_columns_predicted": confusion,
        },
        "oos": {
            "precision": oos_tp / (oos_tp + oos_fp) if oos_tp + oos_fp else 0.0,
            "recall": oos_tp / (oos_tp + oos_fn) if oos_tp + oos_fn else 0.0,
            "false_accept_count": oos_fn,
            "false_reject_count": oos_fp,
        },
    }


def evaluate_legacy_diagnostic(
    artifact_path: Path,
    dataset_path: Path = LEGACY_TEST,
) -> dict[str, Any]:
    if not artifact_path.is_file():
        raise IntentModelError(f"Configured NLU artifact missing: {artifact_path}")
    rows = _load_jsonl(dataset_path)
    if not rows:
        raise ValueError(f"No labeled examples in {dataset_path}")
    engine = NLUEngine.load(path=artifact_path)
    predictions: list[NLUResult] = [engine.parse(row["text"]) for _, row in rows]
    labels = list(load_nlu().get("classifier_labels", []))
    gold = [str(row["intent"]) for _, row in rows]
    predicted = [result.intent for result in predictions]
    scores = _metrics(gold, predicted, labels)

    oos_rows = [
        (row, result)
        for (_, row), result in zip(rows, predictions)
        if row["intent"] == "UNSUPPORTED"
    ]
    rejected = sum(
        validate(result).status.value == "UNSUPPORTED" for _, result in oos_rows
    )
    provenance_path = artifact_path.with_suffix(artifact_path.suffix + ".provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
    return {
        "schema_version": 1,
        "evaluation_role": "legacy_diagnostic_not_independent_test",
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
        "dataset_sha256": sha256_file(dataset_path),
        "n_examples": len(rows),
        "label_counts": dict(Counter(gold)),
        "artifact_path": str(artifact_path.relative_to(PROJECT_ROOT)),
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_training_dataset_sha256": provenance.get("dataset_sha256"),
        "training_manifest_sha256": provenance.get("dataset_manifest_sha256"),
        "model_selection_used_this_dataset": False,
        "metrics": scores,
        "oos_validation": {
            "gold_unsupported_rows": len(oos_rows),
            "rejected_as_unsupported_by_validator": rejected,
            "invalid_tasks_accepted_by_validator": len(oos_rows) - rejected,
        },
        "limitations": [
            "This is the frozen legacy hand-authored dataset, scored as a diagnostic only.",
            "Legacy author/source and template IDs are unavailable; source/template independence is unverified.",
            "It must not be described as an untouched independent test or real-world generalization evidence.",
            "No separate new dev, in-domain, unseen-template, paraphrase, or OOS evaluation set exists yet.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        help="Artifact path; defaults to config/nlu.yaml tfidf_model_path.",
    )
    parser.add_argument("--dataset", type=Path, default=LEGACY_TEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--role",
        choices=("legacy_diagnostic",),
        default="legacy_diagnostic",
        help="Only the explicitly non-independent legacy diagnostic is available now.",
    )
    args = parser.parse_args(argv)
    configured = load_nlu().get("tfidf_model_path")
    artifact = args.artifact or Path(configured or "models/nlu/intent_tfidf_lr.joblib")
    if not artifact.is_absolute():
        artifact = PROJECT_ROOT / artifact
    report = evaluate_legacy_diagnostic(artifact, args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "report": str(args.output.relative_to(PROJECT_ROOT)),
                "evaluation_role": report["evaluation_role"],
                "n_examples": report["n_examples"],
                "artifact_sha256": report["artifact_sha256"],
                "dataset_sha256": report["dataset_sha256"],
                "metrics": report["metrics"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
