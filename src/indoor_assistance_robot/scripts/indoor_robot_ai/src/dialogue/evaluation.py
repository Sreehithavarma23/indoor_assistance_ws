"""Evaluate multi-turn dialogue on a hand-authored development set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.config_loader import PROJECT_ROOT
from src.dialogue.dialogue_manager import DialogueManager
from src.logging_utils import get_logger, setup_logging
from src.nlu.nlu import NLUEngine

logger = get_logger("dialogue.eval")
DATA_PATH = PROJECT_ROOT / "data" / "dialogue" / "eval.jsonl"


def load_scenarios(path: Path = DATA_PATH) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dialogue eval file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append(json.loads(stripped))
    return rows


def run_scenario(engine: NLUEngine, scenario: Dict[str, Any]) -> Dict[str, Any]:
    manager = DialogueManager(engine=engine)
    turns = scenario["turns"]
    last = None
    for utterance in turns:
        last = manager.process(utterance)
    assert last is not None
    expected_status = scenario["expected_status"]
    expected_intent = scenario.get("expected_intent")
    expected_entities = scenario.get("expected_entities") or {}
    status_ok = last.status.value == expected_status
    intent_ok = True
    if expected_intent is not None:
        intent_ok = last.task.get("intent") == expected_intent
    entities_ok = True
    for key, value in expected_entities.items():
        if last.task.get("entities", {}).get(key) != value:
            entities_ok = False
            break
    success = status_ok and intent_ok and entities_ok
    return {
        "id": scenario.get("id"),
        "success": success,
        "expected_status": expected_status,
        "pred_status": last.status.value,
        "expected_intent": expected_intent,
        "pred_intent": last.task.get("intent"),
        "expected_entities": expected_entities,
        "pred_entities": last.task.get("entities"),
        "clarification_count": last.clarification_count,
        "turns": turns,
    }


def evaluate_default() -> Dict[str, Any]:
    engine = NLUEngine.load()
    scenarios = load_scenarios()
    items = [run_scenario(engine, row) for row in scenarios]
    n_ok = sum(1 for item in items if item["success"])
    return {
        "n_scenarios": len(items),
        "scenario_success_rate": n_ok / len(items) if items else 0.0,
        "n_success": n_ok,
        "items": items,
        "note": (
            "Hand-authored multi-turn development set. "
            "Not a real-world dialogue benchmark."
        ),
    }


def main(argv: List[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Evaluate Phase 5 dialogue.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "dialogue" / "eval.json"),
    )
    args = parser.parse_args(argv)
    report = evaluate_default()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_scenarios": report["n_scenarios"],
                "scenario_success_rate": report["scenario_success_rate"],
                "n_success": report["n_success"],
                "note": report["note"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
