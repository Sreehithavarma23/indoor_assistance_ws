"""CLI for the Phase 8 unified AI pipeline.

  PYTHONPATH= python -m src.pipeline --text "Take me to the kitchen" --mode deterministic
  PYTHONPATH= python -m src.pipeline --text "Take me to the kitchen" --mode llm --fake-response plan.json
  PYTHONPATH= python -m src.pipeline --audio path/to/file.wav --mode deterministic --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config_loader import load_llm_planning
from src.llm_planning.provider import FakeLLMProvider, OpenAICompatibleProvider
from src.logging_utils import setup_logging
from src.nlu.nlu import NLUEngine
from src.pipeline.orchestrator import AIPipeline
from src.pipeline.types import PipelineStatus


def _print_human(result) -> None:
    print(f"status: {result.status.value}")
    print(f"input_source: {result.input_source}")
    print(f"input_text: {result.input_text}")
    if result.dialogue_status:
        print(f"dialogue_status: {result.dialogue_status}")
    if result.validation_status:
        print(f"validation_status: {result.validation_status}")
    if result.intent:
        print(f"intent: {result.intent}")
    if result.entities:
        print(f"entities: {result.entities}")
    if result.clarification_question:
        print(f"question: {result.clarification_question}")
    if result.planning_mode:
        print(f"planning_mode: {result.planning_mode.value}")
    if result.planning_status:
        print(f"planning_status: {result.planning_status.value}")
    if result.planner_provider:
        print(f"planner_provider: {result.planner_provider}")
    if result.fallback_used:
        print(f"fallback_used: {result.fallback_used}")
    if result.message:
        print(f"message: {result.message}")
    if result.errors:
        print("errors:")
        for err in result.errors:
            print(f"  - {err}")
    if result.warnings:
        print("warnings:")
        for warn in result.warnings:
            print(f"  - {warn}")
    if result.plan and result.plan.actions:
        print("actions:")
        for action in result.plan.actions:
            deps = f" depends_on={action.depends_on}" if action.depends_on else ""
            print(f"  {action.order}. {action.type.value} {action.parameters}{deps}")
    timings = result.timings.to_dict()
    measured = {k: v for k, v in timings.items() if v is not None}
    if measured:
        print(f"timings_ms: {measured}")


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(
        description=(
            "Phase 8 unified AI pipeline. Symbolic plans only — no robot execution."
        )
    )
    parser.add_argument("--text", help="Natural-language command text.")
    parser.add_argument("--audio", help="Path to an audio file for ASR.")
    parser.add_argument(
        "--mode",
        choices=("deterministic", "llm"),
        default="deterministic",
        help="Planning mode (default: deterministic).",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON result.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use openai_compatible LLM provider (requires API key). LLM mode only.",
    )
    parser.add_argument(
        "--fake-response",
        help="JSON file used as FakeLLMProvider response (offline LLM mode).",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Train Phase 3 intent model from data/nlu/train.jsonl.",
    )
    args = parser.parse_args(argv)

    if bool(args.text) == bool(args.audio):
        parser.error("Provide exactly one of --text or --audio.")

    if args.mode != "llm" and (args.live or args.fake_response):
        parser.error("--live / --fake-response only apply to --mode llm.")

    engine = NLUEngine.train_default() if args.retrain else NLUEngine.load()

    llm_provider = None
    if args.mode == "llm":
        if args.live:
            config = load_llm_planning()
            llm_provider = OpenAICompatibleProvider(
                api_base_url=str(
                    config.get("api_base_url") or "https://api.openai.com/v1"
                ),
                model=str(config.get("model") or "gpt-4o-mini"),
                api_key_env=str(config.get("api_key_env") or "OPENAI_API_KEY"),
                timeout_seconds=float(config.get("timeout_seconds") or 30),
                temperature=float(config.get("temperature") or 0.0),
            )
        elif args.fake_response:
            response_text = Path(args.fake_response).read_text(encoding="utf-8")
            llm_provider = FakeLLMProvider([response_text])
        else:
            print(
                "ERROR: --mode llm requires --fake-response PATH or --live "
                "(with API credentials). Deterministic mode needs no key.",
                file=sys.stderr,
            )
            return 2

    pipeline = AIPipeline(engine=engine, llm_provider=llm_provider)

    if args.text:
        result = pipeline.run_text(args.text, mode=args.mode)
    else:
        result = pipeline.run_audio(args.audio, mode=args.mode)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_human(result)

    return 0 if result.status == PipelineStatus.PLANNED else 1


if __name__ == "__main__":
    raise SystemExit(main())
