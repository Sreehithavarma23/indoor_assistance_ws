"""Hand-authored planning development evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.config_loader import PROJECT_ROOT
from src.logging_utils import get_logger, setup_logging
from src.planning.task_planner import TaskPlanner
from src.planning.types import PlanningStatus

logger = get_logger("planning.eval")
DATA_PATH = PROJECT_ROOT / "data" / "planning" / "eval.jsonl"


def load_cases(path: Path = DATA_PATH) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Planning eval file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append(json.loads(stripped))
    return rows


def _normalize_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "type": item["type"],
            "parameters": item.get("parameters") or {},
        }
        for item in actions
    ]


def evaluate_default() -> Dict[str, Any]:
    planner = TaskPlanner()
    cases = load_cases()
    items: List[Dict[str, Any]] = []
    for case in cases:
        plan = planner.plan_task(
            intent=case["intent"],
            entities=case.get("entities") or {},
            original_text=case.get("id", ""),
        )
        expected_status = case["expected_status"]
        status_ok = plan.status.value == expected_status
        actions_ok = True
        if "expected_actions" in case:
            predicted = _normalize_actions(
                [action.to_dict() for action in plan.actions]
            )
            expected = _normalize_actions(case["expected_actions"])
            actions_ok = predicted == expected
        success = status_ok and actions_ok
        items.append(
            {
                "id": case.get("id"),
                "success": success,
                "expected_status": expected_status,
                "pred_status": plan.status.value,
                "expected_actions": case.get("expected_actions"),
                "pred_actions": [action.to_dict() for action in plan.actions],
                "errors": plan.errors,
            }
        )

    # Determinism check on one RETRIEVE case
    retrieve = next(
        (case for case in cases if case.get("intent") == "RETRIEVE"),
        None,
    )
    deterministic = True
    if retrieve:
        first = planner.plan_task(
            retrieve["intent"], retrieve.get("entities") or {}
        ).to_dict()
        second = planner.plan_task(
            retrieve["intent"], retrieve.get("entities") or {}
        ).to_dict()
        deterministic = first == second

    n_ok = sum(1 for item in items if item["success"])
    return {
        "n_cases": len(items),
        "n_success": n_ok,
        "case_success_rate": n_ok / len(items) if items else 0.0,
        "deterministic_retrieve_repeat": deterministic,
        "items": items,
        "note": (
            "Hand-authored planning development set. "
            "Not a real-world or robot-execution benchmark."
        ),
    }


def main(argv: List[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Evaluate Phase 6 planning.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "planning" / "eval.json"),
    )
    args = parser.parse_args(argv)
    report = evaluate_default()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_cases": report["n_cases"],
                "n_success": report["n_success"],
                "case_success_rate": report["case_success_rate"],
                "deterministic_retrieve_repeat": report[
                    "deterministic_retrieve_repeat"
                ],
                "note": report["note"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
