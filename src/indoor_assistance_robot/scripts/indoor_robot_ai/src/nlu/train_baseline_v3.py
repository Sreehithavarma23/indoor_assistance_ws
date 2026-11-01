"""Train and record the versioned TF-IDF baseline without touching v1/v2."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import yaml
import sklearn

from src.config_loader import PROJECT_ROOT, load_nlu
from src.nlu.baseline_dataset import DATASET_DIR, sha256_file
from src.nlu.intent_classifier import train_intent_classifier

DEFAULT_ARTIFACT = PROJECT_ROOT / "models/nlu/baseline_v3/intent_tfidf_lr.joblib"
DEFAULT_SEED = 42
TRAIN_COMMAND = "PYTHONPATH= .venv/bin/python -m src.nlu.train_baseline_v3"


def _load_training_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "legacy_test" in (row.get("tags") or []):
            raise ValueError(f"legacy_test row found in training data at {path}:{line_number}")
        if not row.get("row_id") or not row.get("text") or not row.get("intent"):
            raise ValueError(f"Missing row_id/text/intent at {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"No training rows in {path}")
    return rows


def _package_versions() -> dict[str, str]:
    names = ("scikit-learn", "numpy", "scipy", "joblib", "PyYAML")
    return {name: importlib.metadata.version(name) for name in names}


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def train_and_save(
    dataset_path: Path,
    artifact_path: Path,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    metadata_path = artifact_path.with_suffix(artifact_path.suffix + ".provenance.json")
    if artifact_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite baseline artifact or provenance: {artifact_path}"
        )

    rows = _load_training_rows(dataset_path)
    labels = load_nlu().get("classifier_labels", [])
    observed = {row["intent"] for row in rows}
    if observed != set(labels):
        raise ValueError(
            f"Training labels differ from configured labels: observed={sorted(observed)}, "
            f"configured={sorted(labels)}"
        )
    texts = [str(row["text"]).strip() for row in rows]
    intents = [str(row["intent"]).strip() for row in rows]
    classifier = train_intent_classifier(texts, intents, random_state=seed)
    classifier.labels = list(labels)

    training_metadata: dict[str, Any] = {
        "schema_version": 1,
        "model_name": "tfidf_logistic_regression_baseline_v3",
        "training_command": TRAIN_COMMAND + f" --seed {seed}",
        "random_seed": seed,
        "dataset_path": _display_path(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_manifest_sha256": sha256_file(DATASET_DIR / "manifest.json"),
        "dataset_exclusion_manifest_sha256": sha256_file(
            DATASET_DIR / "legacy_test_exclusion.json"
        ),
        "training_rows": len(rows),
        "labels": list(labels),
        "preprocessing": {
            "outer_whitespace": "strip at fit and predict",
            "vectorizer": {
                "analyzer": "word",
                "ngram_range": [1, 2],
                "lowercase": True,
                "min_df": 1,
                "strip_accents": "unicode",
                "stop_words": None,
            },
        },
        "model_parameters": {
            "class": "sklearn.linear_model.LogisticRegression",
            "max_iter": 1000,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "random_state": seed,
        },
        "package_versions": _package_versions(),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "config_nlu_sha256": hashlib.sha256(
            yaml.safe_dump(load_nlu(), sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    classifier.training_metadata = training_metadata
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(artifact_path, metadata=training_metadata)
    provenance = {
        **training_metadata,
        "artifact_path": _display_path(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
    }
    with metadata_path.open("x", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR / "train.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    metadata = train_and_save(args.dataset, args.output, args.seed)
    print(
        json.dumps(
            {
                "artifact": metadata["artifact_path"],
                "artifact_sha256": metadata["artifact_sha256"],
                "dataset_sha256": metadata["dataset_sha256"],
                "training_rows": metadata["training_rows"],
                "seed": metadata["random_seed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
