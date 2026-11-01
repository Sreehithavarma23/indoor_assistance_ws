"""Hand-authored offline evaluation for Phase 7 LLM-assisted planning.

Uses FakeLLMProvider by default. Does not call paid APIs unless --live is set.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config_loader import PROJECT_ROOT, load_llm_planning
from src.dialogue import DialogueManager
from src.llm_planning.planner import LLMAssistedPlanner
from src.llm_planning.provider import FakeLLMProvider, OpenAICompatibleProvider
from src.logging_utils import setup_logging
from src.nlu.nlu import NLUEngine
from src.planning.task_planner import TaskPlanner

DATA_PATH = PROJECT_ROOT / "data" / "llm_planning" / "eval.jsonl"


def load_cases(path: Path = DATA_PATH) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"LLM planning eval file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append(json.loads(stripped))
    return rows


def _action_types(plan) -> List[str]:
    return [action.type.value for action in plan.actions]


def evaluate_default(
    *,
    live: bool = False,
    engine: Optional[NLUEngine] = None,
) -> Dict[str, Any]:
    engine = engine or NLUEngine.load()
    deterministic = TaskPlanner()
    cases = load_cases()
    items: List[Dict[str, Any]] = []
    live_calls = 0

    for case in cases:
        # Fresh dialogue state per case so pending clarifications do not leak.
        manager = DialogueManager(engine=engine)
        text = case["text"]
        dialogue = manager.process(text)
        dialogue_ok = dialogue.status.value == case["expect_dialogue_status"]

        det_plan = None
        det_types = None
        if case.get("compare_deterministic") and dialogue.status.value == "READY":
            det_plan = deterministic.plan(dialogue)
            det_types = _action_types(det_plan)

        llm_result = None
        llm_ok = False
        mode = None
        fallback_used = False
        latency_ms = None
        parse_errors: List[str] = []

        if case.get("skip_llm_call"):
            # Still exercise gating via planner with a dummy provider.
            planner = LLMAssistedPlanner(
                provider=FakeLLMProvider(["{}"]),
                max_correction_attempts=0,
                fallback_to_deterministic=False,
            )
            llm_result = planner.plan(dialogue)
            llm_ok = llm_result.plan.status.value == case["expected_llm_status"]
            mode = llm_result.mode.value
            fallback_used = llm_result.fallback_used
            latency_ms = llm_result.latency_ms
            parse_errors = list(llm_result.parse_errors)
        else:
            if live:
                config = load_llm_planning()
                provider = OpenAICompatibleProvider(
                    api_base_url=str(
                        config.get("api_base_url") or "https://api.openai.com/v1"
                    ),
                    model=str(config.get("model") or "gpt-4o-mini"),
                    api_key_env=str(config.get("api_key_env") or "OPENAI_API_KEY"),
                    timeout_seconds=float(config.get("timeout_seconds") or 30),
                    temperature=float(config.get("temperature") or 0.0),
                )
                live_calls += 1
            else:
                response = case.get("fake_llm_response")
                if response is None:
                    raise ValueError(
                        f"Case {case.get('id')} missing fake_llm_response "
                        "for offline evaluation."
                    )
                provider = FakeLLMProvider([json.dumps(response)])

            planner = LLMAssistedPlanner(
                provider=provider,
                max_correction_attempts=0 if case.get("expect_rejection") else 1,
                fallback_to_deterministic=False,
            )
            started = time.perf_counter()
            llm_result = planner.plan(dialogue)
            latency_ms = (time.perf_counter() - started) * 1000.0
            mode = llm_result.mode.value
            fallback_used = llm_result.fallback_used
            parse_errors = list(llm_result.parse_errors)

            status_ok = (
                llm_result.plan.status.value == case["expected_llm_status"]
            )
            types_ok = True
            if "expected_action_types" in case:
                types_ok = (
                    _action_types(llm_result.plan) == case["expected_action_types"]
                )
            llm_ok = status_ok and types_ok

        structural_match = None
        if (
            det_types is not None
            and llm_result is not None
            and llm_result.plan.status.value == "PLANNED"
            and case.get("compare_deterministic")
        ):
            structural_match = _action_types(llm_result.plan) == det_types

        success = dialogue_ok and llm_ok
        items.append(
            {
                "id": case.get("id"),
                "success": success,
                "dialogue_status": dialogue.status.value,
                "expect_dialogue_status": case["expect_dialogue_status"],
                "deterministic_action_types": det_types,
                "llm_status": None
                if llm_result is None
                else llm_result.plan.status.value,
                "llm_action_types": None
                if llm_result is None
                else _action_types(llm_result.plan),
                "structural_match_deterministic": structural_match,
                "mode": mode,
                "fallback_used": fallback_used,
                "latency_ms": latency_ms,
                "parse_errors": parse_errors,
            }
        )

    n_ok = sum(1 for item in items if item["success"])
    comparable = [
        item
        for item in items
        if item.get("structural_match_deterministic") is not None
    ]
    n_match = sum(
        1 for item in comparable if item["structural_match_deterministic"]
    )
    return {
        "n_cases": len(items),
        "n_success": n_ok,
        "case_success_rate": n_ok / len(items) if items else 0.0,
        "n_structural_compare": len(comparable),
        "n_structural_match_deterministic": n_match,
        "live_provider_used": live,
        "live_provider_calls": live_calls,
        "note": (
            "Hand-authored offline development set with FakeLLMProvider. "
            "Not a live-model or robot-execution benchmark."
            if not live
            else "Includes live provider calls. Interpret cautiously; small N."
        ),
        "items": items,
    }


def main(argv: List[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Evaluate Phase 7 LLM planning.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "llm_planning" / "eval.json"),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call the configured openai_compatible provider (needs API key).",
    )
    args = parser.parse_args(argv)
    report = evaluate_default(live=bool(args.live))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        "n_cases": report["n_cases"],
        "n_success": report["n_success"],
        "case_success_rate": report["case_success_rate"],
        "n_structural_compare": report["n_structural_compare"],
        "n_structural_match_deterministic": report[
            "n_structural_match_deterministic"
        ],
        "live_provider_used": report["live_provider_used"],
        "live_provider_calls": report["live_provider_calls"],
        "note": report["note"],
    }
    print(json.dumps(summary, indent=2))
    return 0 if report["n_success"] == report["n_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
