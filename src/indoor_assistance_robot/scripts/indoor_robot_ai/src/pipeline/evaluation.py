"""Offline end-to-end evaluation for the Phase 8 pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config_loader import PROJECT_ROOT
from src.llm_planning.exceptions import ProviderError
from src.llm_planning.planner import LLMAssistedPlanner
from src.llm_planning.provider import FakeLLMProvider
from src.logging_utils import setup_logging
from src.nlu.nlu import NLUEngine
from src.pipeline.orchestrator import AIPipeline
from src.pipeline.types import PipelineStatus
from src.planning.types import PlanningStatus

DATA_PATH = PROJECT_ROOT / "data" / "pipeline" / "eval.jsonl"


def load_cases(path: Path = DATA_PATH) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline eval file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append(json.loads(stripped))
    return rows


def _action_types(result) -> Optional[List[str]]:
    if result.plan is None:
        return None
    return [action.type.value for action in result.plan.actions]


def evaluate_default(engine: Optional[NLUEngine] = None) -> Dict[str, Any]:
    engine = engine or NLUEngine.load()
    cases = load_cases()
    items: List[Dict[str, Any]] = []

    n_det = 0
    n_llm = 0
    n_intent_ok = 0
    n_intent_labeled = 0
    n_dialogue_ok = 0
    n_dialogue_labeled = 0
    n_plan_seq_ok = 0
    n_plan_seq_labeled = 0
    latencies: List[float] = []

    for case in cases:
        mode = str(case.get("mode") or "deterministic")
        if mode == "llm":
            n_llm += 1
            if case.get("provider_error"):
                provider = FakeLLMProvider(fail_with=ProviderError("simulated"))
            else:
                provider = FakeLLMProvider(
                    [json.dumps(case["fake_llm_response"])]
                )
            pipeline = AIPipeline(
                engine=engine,
                llm_planner=LLMAssistedPlanner(
                    provider=provider,
                    max_correction_attempts=0,
                    fallback_to_deterministic=False,
                ),
            )
        else:
            n_det += 1
            pipeline = AIPipeline(engine=engine)

        session = pipeline.new_session()
        result = None
        for turn in case["turns"]:
            result = pipeline.run_text(turn, session=session, mode=mode)
        assert result is not None

        if result.timings.total_ms is not None:
            latencies.append(result.timings.total_ms)

        status_ok = result.status.value == case["expected_final_status"]

        dialogue_ok = True
        if "expected_dialogue_status" in case:
            n_dialogue_labeled += 1
            dialogue_ok = result.dialogue_status == case["expected_dialogue_status"]
            if dialogue_ok:
                n_dialogue_ok += 1

        intent_ok = True
        if "expected_intent" in case:
            n_intent_labeled += 1
            intent_ok = result.intent == case["expected_intent"]
            if intent_ok:
                n_intent_ok += 1

        plan_ok = True
        if case.get("expect_no_plan") or case.get("expect_no_successful_plan"):
            plan_ok = not (
                result.plan is not None
                and result.planning_status == PlanningStatus.PLANNED
            )
        if "expected_action_types" in case:
            n_plan_seq_labeled += 1
            types = _action_types(result) or []
            plan_ok = types == case["expected_action_types"]
            if plan_ok:
                n_plan_seq_ok += 1

        dest_ok = True
        if "expected_destination" in case and result.plan and result.plan.actions:
            dest_ok = (
                result.plan.actions[0].parameters.get("destination")
                == case["expected_destination"]
            )

        mode_ok = True
        if "expected_planning_mode" in case:
            mode_ok = (
                result.planning_mode is not None
                and result.planning_mode.value == case["expected_planning_mode"]
            )

        success = status_ok and dialogue_ok and intent_ok and plan_ok and dest_ok and mode_ok
        items.append(
            {
                "id": case.get("id"),
                "success": success,
                "mode": mode,
                "final_status": result.status.value,
                "dialogue_status": result.dialogue_status,
                "intent": result.intent,
                "action_types": _action_types(result),
                "planning_mode": (
                    result.planning_mode.value if result.planning_mode else None
                ),
                "total_ms": result.timings.total_ms,
            }
        )

    n_ok = sum(1 for item in items if item["success"])
    return {
        "n_cases": len(items),
        "n_success": n_ok,
        "end_to_end_success_rate": n_ok / len(items) if items else 0.0,
        "n_deterministic_cases": n_det,
        "n_llm_cases": n_llm,
        "intent_accuracy": (
            n_intent_ok / n_intent_labeled if n_intent_labeled else None
        ),
        "n_intent_labeled": n_intent_labeled,
        "dialogue_status_accuracy": (
            n_dialogue_ok / n_dialogue_labeled if n_dialogue_labeled else None
        ),
        "n_dialogue_labeled": n_dialogue_labeled,
        "plan_sequence_accuracy": (
            n_plan_seq_ok / n_plan_seq_labeled if n_plan_seq_labeled else None
        ),
        "n_plan_sequence_labeled": n_plan_seq_labeled,
        "mean_total_latency_ms": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "live_provider_used": False,
        "note": (
            "Hand-authored offline development set. "
            "Not a real-world ASR/NLU/robot benchmark. "
            "Metrics: end_to_end_success_rate = fraction of cases matching all "
            "labeled expectations; intent_accuracy / dialogue_status_accuracy / "
            "plan_sequence_accuracy are labeled subsets only."
        ),
        "items": items,
    }


def main(argv: List[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Evaluate Phase 8 pipeline.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "pipeline" / "eval.json"),
    )
    args = parser.parse_args(argv)
    report = evaluate_default()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        key: report[key]
        for key in (
            "n_cases",
            "n_success",
            "end_to_end_success_rate",
            "n_deterministic_cases",
            "n_llm_cases",
            "intent_accuracy",
            "dialogue_status_accuracy",
            "plan_sequence_accuracy",
            "mean_total_latency_ms",
            "live_provider_used",
            "note",
        )
    }
    print(json.dumps(summary, indent=2))
    return 0 if report["n_success"] == report["n_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
