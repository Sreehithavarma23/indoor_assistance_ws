"""End-to-end dry evaluation on frozen NLU v2 text splits (no ASR, no ROS)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.config_loader import PROJECT_ROOT
from src.data_pipeline.split import load_split
from src.logging_utils import setup_logging
from src.pipeline.orchestrator import AIPipeline
from src.pipeline.types import PipelineStatus

RESULTS = PROJECT_ROOT / "results" / "nlu_v2" / "e2e_frozen.json"

# Statuses that must never produce an executable plan.
_BLOCKED = {
    PipelineStatus.UNSUPPORTED,
    PipelineStatus.CLARIFICATION_REQUIRED,
    PipelineStatus.CLARIFICATION_FAILED,
    PipelineStatus.CANCELLED,
    PipelineStatus.REJECTED,
    PipelineStatus.FAILED,
}


def evaluate(split_name: str, pipeline: AIPipeline) -> Dict[str, Any]:
    rows = load_split(split_name)
    intent_ok = 0
    plan_when_supported = 0
    supported = 0
    invalid_plans = 0
    oos_rejected = 0
    oos_total = 0
    for row in rows:
        session = pipeline.new_session()
        result = pipeline.run_text(row.text, session=session, mode="deterministic")
        if result.intent == row.intent:
            intent_ok += 1
        if row.intent == "UNSUPPORTED":
            oos_total += 1
            if result.status in _BLOCKED and result.plan is None:
                oos_rejected += 1
            if result.status == PipelineStatus.PLANNED:
                invalid_plans += 1
        else:
            supported += 1
            if result.status == PipelineStatus.PLANNED and result.plan is not None:
                plan_when_supported += 1
            # Incomplete/ambiguous commands may correctly ask for clarification.
    return {
        "split": split_name,
        "n": len(rows),
        "intent_accuracy": intent_ok / len(rows) if rows else 0.0,
        "supported_n": supported,
        "planned_among_supported": plan_when_supported,
        "oos_n": oos_total,
        "oos_blocked_without_plan": oos_rejected,
        "invalid_execution_attempts": invalid_plans,
        "note": (
            "Text-only e2e using the default TF-IDF NLUEngine + deterministic planner. "
            "Missing-slot commands are expected to clarify rather than plan."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test_indomain", "test_paraphrase", "test_oos"],
    )
    parser.add_argument("--output", default=str(RESULTS))
    args = parser.parse_args(argv)
    pipeline = AIPipeline()
    payload = {"splits": {}}
    for name in args.splits:
        payload["splits"][name] = evaluate(name, pipeline)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
