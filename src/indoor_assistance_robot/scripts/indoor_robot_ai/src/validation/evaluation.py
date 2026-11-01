"""Evaluate validation statuses on a hand-authored development set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from src.config_loader import PROJECT_ROOT
from src.logging_utils import get_logger, setup_logging
from src.nlu.nlu import NLUEngine
from src.validation.validator import validate

logger = get_logger("validation.eval")
DATA_PATH = PROJECT_ROOT / "data" / "validation" / "eval.jsonl"


def load_eval_examples(path: Path = DATA_PATH) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Validation eval file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append(json.loads(stripped))
    return rows


def evaluate_default() -> Dict[str, Any]:
    engine = NLUEngine.load()
    examples = load_eval_examples()
    gold = [row["status"] for row in examples]
    pred: List[str] = []
    items: List[Dict[str, Any]] = []
    for row in examples:
        outcome = validate(engine.parse(row["text"]))
        pred.append(outcome.status.value)
        items.append(
            {
                "text": row["text"],
                "gold_status": row["status"],
                "pred_status": outcome.status.value,
                "intent": outcome.intent,
                "entities": outcome.entities,
                "missing_entities": outcome.missing_entities,
                "ambiguous_entities": outcome.ambiguous_entities,
                "correct": outcome.status.value == row["status"],
            }
        )

    labels = sorted(set(gold) | set(pred))
    confusion = {row_l: {col: 0 for col in labels} for row_l in labels}
    for true, guessed in zip(gold, pred):
        confusion[true][guessed] += 1
    correct = sum(1 for item in items if item["correct"])
    return {
        "n_examples": len(examples),
        "status_accuracy": correct / len(examples) if examples else 0.0,
        "confusion": confusion,
        "gold_counts": dict(Counter(gold)),
        "items": items,
        "note": (
            "Hand-authored development set. These scores are not a real-world "
            "benchmark and must not be reported as deployment accuracy."
        ),
    }


def main(argv: List[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Evaluate Phase 4 validation.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "validation" / "eval.json"),
    )
    args = parser.parse_args(argv)
    report = evaluate_default()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_examples": report["n_examples"],
                "status_accuracy": report["status_accuracy"],
                "gold_counts": report["gold_counts"],
                "note": report["note"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
