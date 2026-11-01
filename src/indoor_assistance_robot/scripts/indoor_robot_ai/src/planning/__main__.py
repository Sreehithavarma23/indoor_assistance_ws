"""CLI: plan a READY command through the existing dialogue pipeline.

  PYTHONPATH= python -m src.planning "Take me to the kitchen."
  PYTHONPATH= python -m src.planning "Bring me the bottle from the kitchen." --json
"""

from __future__ import annotations

import argparse
import json

from src.dialogue import DialogueManager
from src.logging_utils import setup_logging
from src.nlu.nlu import NLUEngine
from src.planning.task_planner import TaskPlanner


def _print_human(plan) -> None:
    print(f"status: {plan.status.value}")
    print(f"intent: {plan.intent}")
    if plan.explanation:
        print(f"explanation: {plan.explanation}")
    if plan.errors:
        print("errors:")
        for err in plan.errors:
            print(f"  - {err}")
    if plan.warnings:
        print("warnings:")
        for warn in plan.warnings:
            print(f"  - {warn}")
    if plan.actions:
        print("actions:")
        for action in plan.actions:
            deps = f" depends_on={action.depends_on}" if action.depends_on else ""
            params = action.parameters or {}
            print(
                f"  {action.order}. {action.type.value} {params}{deps}"
            )


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Phase 6 planner. Symbolic plans only — no robot execution."
    )
    parser.add_argument("text", nargs="+", help="Command text (quote the sentence).")
    parser.add_argument("--json", action="store_true", help="Print JSON plan.")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Train Phase 3 intent model from data/nlu/train.jsonl.",
    )
    args = parser.parse_args(argv)
    text = " ".join(args.text)

    engine = NLUEngine.train_default() if args.retrain else NLUEngine.load()
    dialogue = DialogueManager(engine=engine).process(text)
    plan = TaskPlanner().plan(dialogue)

    if args.json:
        print(
            json.dumps(
                {
                    "dialogue_status": dialogue.status.value,
                    "dialogue_task": dialogue.task,
                    "plan": plan.to_dict(),
                },
                indent=2,
            )
        )
    else:
        print(f"dialogue_status: {dialogue.status.value}")
        if dialogue.status.value != "READY":
            print(f"dialogue_message: {dialogue.message or dialogue.question}")
        _print_human(plan)
    return 0 if plan.status.value == "PLANNED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
