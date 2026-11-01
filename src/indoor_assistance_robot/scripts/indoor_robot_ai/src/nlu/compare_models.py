"""Train and compare TF-IDF vs embedding intent classifiers on NLU v2 splits.

Uses train for fitting, dev for selection notes, frozen tests for final numbers.
Never tunes thresholds on frozen test sets in this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from src.config_loader import PROJECT_ROOT
from src.data_pipeline.split import load_split
from src.logging_utils import get_logger, setup_logging
from src.nlu.embedding_classifier import (
    DEFAULT_ARTIFACT_DIR,
    EmbeddingLogRegIntentClassifier,
    embedding_available,
    train_embedding_classifier,
)
from src.nlu.intent_classifier import (
    TfidfLogRegIntentClassifier,
    default_model_path,
    train_intent_classifier,
)

logger = get_logger("nlu.compare")
RESULTS_DIR = PROJECT_ROOT / "results" / "nlu_v2"


def _fingerprint(texts: Sequence[str], labels: Sequence[str]) -> str:
    h = hashlib.sha256()
    for text, label in zip(texts, labels):
        h.update(text.encode("utf-8"))
        h.update(b"\t")
        h.update(label.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def _scores(gold: Sequence[str], pred: Sequence[str]) -> Dict[str, Any]:
    labels = sorted(set(gold) | set(pred))
    correct = sum(1 for t, p in zip(gold, pred) if t == p)
    accuracy = correct / len(gold) if gold else 0.0
    per_class: Dict[str, Dict[str, float]] = {}
    f1s = []
    for label in labels:
        tp = sum(1 for t, p in zip(gold, pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(gold, pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(gold, pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        support = sum(1 for t in gold if t == label)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        if support:
            f1s.append(f1)
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "rows_gold_columns_predicted": [
                [
                    sum(1 for actual, guessed in zip(gold, pred) if actual == row and guessed == col)
                    for col in labels
                ]
                for row in labels
            ],
        },
        "n": len(gold),
        "label_counts": dict(Counter(gold)),
    }


def _oos_metrics(gold: Sequence[str], pred: Sequence[str]) -> Dict[str, float]:
    """UNSUPPORTED as the positive rejection class."""
    tp = sum(1 for t, p in zip(gold, pred) if t == "UNSUPPORTED" and p == "UNSUPPORTED")
    fp = sum(1 for t, p in zip(gold, pred) if t != "UNSUPPORTED" and p == "UNSUPPORTED")
    fn = sum(1 for t, p in zip(gold, pred) if t == "UNSUPPORTED" and p != "UNSUPPORTED")
    # False acceptance = supported gold predicted as non-UNSUPPORTED incorrectly?
    # Spec: false acceptance of unsupported = predicting a robot intent when gold is UNSUPPORTED.
    false_accept = fn  # gold UNSUPPORTED, pred not
    false_reject = fp  # gold supported, pred UNSUPPORTED
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "oos_precision": precision,
        "oos_recall": recall,
        "false_acceptance_of_unsupported": false_accept,
        "false_rejection_of_supported": false_reject,
    }


def _predict_all(predict_fn, texts: Sequence[str]) -> Tuple[List[str], float]:
    started = time.perf_counter()
    preds = [predict_fn(text).intent for text in texts]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return preds, elapsed_ms


def evaluate_classifier(name: str, predict_fn, splits: Dict[str, Any]) -> Dict[str, Any]:
    report: Dict[str, Any] = {"model": name, "splits": {}}
    for split_name, rows in splits.items():
        texts = [r.text for r in rows]
        gold = [r.intent for r in rows]
        preds, latency_ms = _predict_all(predict_fn, texts)
        metrics = _scores(gold, preds)
        metrics.update(_oos_metrics(gold, preds))
        metrics["latency_ms_total"] = latency_ms
        metrics["latency_ms_per_example"] = latency_ms / len(texts) if texts else 0.0
        report["splits"][split_name] = metrics
    return report


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Compare intent classifiers on NLU v2.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "model_comparison.json"),
    )
    args = parser.parse_args(argv)

    train = load_split("train")
    dev = load_split("dev")
    frozen = {
        "dev": dev,
        "test_indomain": load_split("test_indomain"),
        "test_paraphrase": load_split("test_paraphrase"),
        "test_oos": load_split("test_oos"),
    }
    train_texts = [r.text for r in train]
    train_labels = [r.intent for r in train]
    fingerprint = _fingerprint(train_texts, train_labels)

    tfidf = train_intent_classifier(
        train_texts, train_labels, random_state=args.seed
    )
    baseline_path = PROJECT_ROOT / "models" / "nlu" / "intent_tfidf_lr_v2.joblib"
    tfidf.save(baseline_path)

    reports = []
    reports.append(
        evaluate_classifier(
            "tfidf_logreg",
            tfidf.predict,
            frozen,
        )
    )

    embedding_meta = {
        "available": embedding_available(),
        "trained": False,
        "encoder": None,
        "fine_tuned_encoder": False,
    }
    if not args.skip_embedding and embedding_available():
        emb = train_embedding_classifier(
            train_texts, train_labels, random_state=args.seed
        )
        emb.save(DEFAULT_ARTIFACT_DIR)
        embedding_meta.update(
            {
                "trained": True,
                "encoder": emb.encoder_name,
                "artifact_dir": str(DEFAULT_ARTIFACT_DIR),
            }
        )
        reports.append(
            evaluate_classifier(
                "minilm_logreg",
                emb.predict,
                frozen,
            )
        )
    else:
        logger.warning("Skipping embedding classifier (unavailable or --skip-embedding).")

    # Dev-only model selection note (not frozen-test tuning).
    selection = {}
    for report in reports:
        selection[report["model"]] = {
            "dev_accuracy": report["splits"]["dev"]["accuracy"],
            "dev_macro_f1": report["splits"]["dev"]["macro_f1"],
        }
    selected = max(selection, key=lambda m: selection[m]["dev_macro_f1"])

    payload = {
        "dataset_fingerprint_train": fingerprint,
        "n_train": len(train),
        "random_seed": args.seed,
        "baseline_legacy_model_path": str(default_model_path()),
        "v2_tfidf_model_path": str(baseline_path),
        "embedding": embedding_meta,
        "dev_selection": selection,
        "selected_by_dev_macro_f1": selected,
        "note": (
            "Frozen test splits were not used for threshold tuning. "
            "Selected model is by development macro-F1 only."
        ),
        "models": reports,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(out),
        "selected_by_dev_macro_f1": selected,
        "dev": selection,
        "test_indomain": {
            r["model"]: r["splits"]["test_indomain"]["accuracy"] for r in reports
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
