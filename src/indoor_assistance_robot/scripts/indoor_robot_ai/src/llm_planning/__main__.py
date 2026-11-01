"""CLI for Phase 7 constrained LLM-assisted planning.

  # Deterministic baseline (Phase 6) via this CLI:
  PYTHONPATH= python -m src.llm_planning "Take me to the kitchen." --mode deterministic

  # LLM-assisted with fake/scripted provider (offline default):
  PYTHONPATH= python -m src.llm_planning "Take me to the kitchen." --mode llm --fake-response path.json

  # Live OpenAI-compatible provider (requires OPENAI_API_KEY):
  PYTHONPATH= python -m src.llm_planning "Take me to the kitchen." --mode llm --live --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dialogue import DialogueManager
from src.llm_planning.planner import LLMAssistedPlanner
from src.llm_planning.provider import FakeLLMProvider, OpenAICompatibleProvider
from src.logging_utils import setup_logging
from src.nlu.nlu import NLUEngine
from src.planning.task_planner import TaskPlanner
from src.config_loader import load_llm_planning


def _print_human(result_plan, *, mode: str, meta: dict | None = None) -> None:
    print(f"mode: {mode}")
    if meta:
        for key in ("provider", "attempts", "fallback_used", "latency_ms", "message"):
            if key in meta and meta[key] is not None:
                print(f"{key}: {meta[key]}")
    print(f"status: {result_plan.status.value}")
    print(f"intent: {result_plan.intent}")
    if result_plan.explanation:
        print(f"explanation: {result_plan.explanation}")
    if result_plan.errors:
        print("errors:")
        for err in result_plan.errors:
            print(f"  - {err}")
    if result_plan.warnings:
        print("warnings:")
        for warn in result_plan.warnings:
            print(f"  - {warn}")
    if result_plan.actions:
        print("actions:")
        for action in result_plan.actions:
            deps = f" depends_on={action.depends_on}" if action.depends_on else ""
            print(f"  {action.order}. {action.type.value} {action.parameters}{deps}")


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(
        description=(
            "Phase 7 planner CLI. Symbolic plans only — no robot execution. "
            "Deterministic mode needs no credentials."
        )
    )
    parser.add_argument("text", nargs="+", help="Command text (quote the sentence).")
    parser.add_argument(
        "--mode",
        choices=("deterministic", "llm"),
        default="deterministic",
        help="Planning mode (default: deterministic Phase 6).",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use openai_compatible provider from config (requires API key).",
    )
    parser.add_argument(
        "--fake-response",
        help="Path to a JSON file used as FakeLLMProvider response (offline LLM mode).",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="On LLM failure, fall back to the Phase 6 deterministic planner.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Train Phase 3 intent model from data/nlu/train.jsonl.",
    )
    args = parser.parse_args(argv)
    text = " ".join(args.text)

    engine = NLUEngine.train_default() if args.retrain else NLUEngine.load()
    dialogue = DialogueManager(engine=engine).process(text)

    if args.mode == "deterministic":
        plan = TaskPlanner().plan(dialogue)
        payload = {
            "dialogue_status": dialogue.status.value,
            "dialogue_task": dialogue.task,
            "mode": "DETERMINISTIC",
            "plan": plan.to_dict(),
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"dialogue_status: {dialogue.status.value}")
            if dialogue.status.value != "READY":
                print(f"dialogue_message: {dialogue.message or dialogue.question}")
            _print_human(plan, mode="DETERMINISTIC")
        return 0 if plan.status.value == "PLANNED" else 1

    config = load_llm_planning()
    if args.live:
        provider = OpenAICompatibleProvider(
            api_base_url=str(config.get("api_base_url") or "https://api.openai.com/v1"),
            model=str(config.get("model") or "gpt-4o-mini"),
            api_key_env=str(config.get("api_key_env") or "OPENAI_API_KEY"),
            timeout_seconds=float(config.get("timeout_seconds") or 30),
            temperature=float(config.get("temperature") or 0.0),
        )
    elif args.fake_response:
        response_text = Path(args.fake_response).read_text(encoding="utf-8")
        provider = FakeLLMProvider([response_text])
    else:
        # Default offline: require scripted response or use empty fake (will fail clearly).
        provider = FakeLLMProvider()
        print(
            "NOTE: LLM mode without --live or --fake-response uses an empty "
            "FakeLLMProvider and will fail unless you inject responses in code.",
            flush=True,
        )

    planner = LLMAssistedPlanner(
        provider=provider,
        fallback_to_deterministic=bool(args.fallback),
    )
    result = planner.plan(dialogue)
    payload = {
        "dialogue_status": dialogue.status.value,
        "dialogue_task": dialogue.task,
        "llm_result": result.to_dict(),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"dialogue_status: {dialogue.status.value}")
        if dialogue.status.value != "READY":
            print(f"dialogue_message: {dialogue.message or dialogue.question}")
        _print_human(
            result.plan,
            mode=result.mode.value,
            meta={
                "provider": result.provider,
                "attempts": result.attempts,
                "fallback_used": result.fallback_used,
                "latency_ms": result.latency_ms,
                "message": result.message,
            },
        )
        if result.parse_errors:
            print("parse_errors:")
            for err in result.parse_errors:
                print(f"  - {err}")
    return 0 if result.plan.status.value == "PLANNED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
