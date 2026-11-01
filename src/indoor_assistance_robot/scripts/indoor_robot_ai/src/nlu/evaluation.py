"""Evaluate the trained NLU baseline on the held-out test split.

Reports only numbers computed from this dataset. The set is small and
hand-authored, so the scores are development metrics, not a benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from src.config_loader import PROJECT_ROOT
from src.logging_utils import get_logger, setup_logging
from src.nlu.dataset import DATA_DIR, dataset_summary, load_test
from src.nlu.nlu import NLUEngine
from src.nlu.types import NLUExample, NLUResult

logger = get_logger("nlu.eval")

ENTITY_SLOTS = ("object", "source_location", "destination", "person", "action")


def _confusion(gold: Sequence[str], pred: Sequence[str]) -> Dict[str, Dict[str, int]]:
    labels = sorted(set(gold) | set(pred))
    matrix = {row: {col: 0 for col in labels} for row in labels}
    for true, guessed in zip(gold, pred):
        matrix[true][guessed] += 1
    return matrix


def _per_class_scores(
    gold: Sequence[str],
    pred: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    labels = sorted(set(gold) | set(pred))
    scores: Dict[str, Dict[str, float]] = {}
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
        scores[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": gold.count(label) if isinstance(gold, list) else sum(1 for item in gold if item == label),
        }
    return scores


def _entity_prf(
    gold_examples: Sequence[NLUExample],
    results: Sequence[NLUResult],
) -> Dict[str, Any]:
    tp = fp = fn = 0
    per_slot = {slot: Counter() for slot in ENTITY_SLOTS}
    for example, result in zip(gold_examples, results):
        gold_items = {
            (slot, value)
            for slot, value in example.entities.items()
            if slot in ENTITY_SLOTS
        }
        pred_items = {
            (slot, value)
            for slot, value in result.entities.items()
            if slot in ENTITY_SLOTS
        }
        tp += len(gold_items & pred_items)
        fp += len(pred_items - gold_items)
        fn += len(gold_items - pred_items)
        for slot in ENTITY_SLOTS:
            g = example.entities.get(slot)
            p = result.entities.get(slot)
            if g and p and g == p:
                per_slot[slot]["tp"] += 1
            elif g and g != p:
                per_slot[slot]["fn"] += 1
                if p:
                    per_slot[slot]["fp"] += 1
            elif p and not g:
                per_slot[slot]["fp"] += 1

    def pack(true_pos: int, false_pos: int, false_neg: int) -> Dict[str, float]:
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0.0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": true_pos,
            "fp": false_pos,
            "fn": false_neg,
        }

    slot_scores = {
        slot: pack(counts["tp"], counts["fp"], counts["fn"])
        for slot, counts in per_slot.items()
    }
    return {"micro": pack(tp, fp, fn), "per_slot": slot_scores}


def evaluate_engine(
    engine: NLUEngine,
    examples: Sequence[NLUExample],
) -> Dict[str, Any]:
    results = [engine.parse(example.text) for example in examples]
    gold_intents = [example.intent for example in examples]
    pred_intents = [result.intent for result in results]
    correct = sum(1 for g, p in zip(gold_intents, pred_intents) if g == p)
    report: Dict[str, Any] = {
        "n_examples": len(examples),
        "intent_accuracy": correct / len(examples) if examples else 0.0,
        "intent_per_class": _per_class_scores(gold_intents, pred_intents),
        "intent_confusion": _confusion(gold_intents, pred_intents),
        "entity": _entity_prf(examples, results),
        "errors": [
            {
                "text": example.text,
                "gold_intent": example.intent,
                "pred_intent": result.intent,
                "gold_entities": example.entities,
                "pred_entities": result.entities,
            }
            for example, result in zip(examples, results)
            if example.intent != result.intent or example.entities != result.entities
        ],
        "note": (
            "Hand-authored development set. These scores are not a real-world "
            "benchmark and must not be reported as deployment accuracy."
        ),
    }
    return report


def evaluate_default(engine: Optional[NLUEngine] = None) -> Dict[str, Any]:
    test = load_test()
    engine = engine or NLUEngine.load()
    report = evaluate_engine(engine, test)
    report["test_intents"] = dataset_summary(test)
    artifact_path = getattr(engine.classifier, "artifact_path", None)
    if artifact_path is None:
        from src.config_loader import load_nlu
        from src.nlu.intent_classifier import default_model_path

        configured = load_nlu().get("tfidf_model_path")
        artifact_path = Path(configured) if configured else default_model_path()
        if not artifact_path.is_absolute():
            artifact_path = PROJECT_ROOT / artifact_path
    digest = hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest()
    report.update(
        {
            "evaluation_role": "legacy_diagnostic_not_independent_test",
            "dataset_path": str(DATA_DIR / "test.jsonl"),
            "dataset_sha256": hashlib.sha256(
                (DATA_DIR / "test.jsonl").read_bytes()
            ).hexdigest(),
            "artifact_path": str(Path(artifact_path).relative_to(PROJECT_ROOT)),
            "artifact_sha256": digest,
            "note": (
                "Legacy hand-authored diagnostic scored with the configured saved artifact. "
                "Author/source and template independence are unverified; this is not an "
                "untouched test or real-world benchmark."
            ),
        }
    )
    return report


def write_report(report: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2))
    logger.info("Wrote NLU evaluation to %s", path)
    return path


def main(argv: List[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Evaluate the Phase 3 NLU baseline.")
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT
            / "results"
            / "nlu_baseline_v3"
            / (
                "legacy_diagnostic_detailed_"
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
            )
        ),
    )
    args = parser.parse_args(argv)
    report = evaluate_default()
    write_report(report, Path(args.output))
    summary = {
        "n_examples": report["n_examples"],
        "intent_accuracy": report["intent_accuracy"],
        "entity_micro_f1": report["entity"]["micro"]["f1"],
        "note": report["note"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
