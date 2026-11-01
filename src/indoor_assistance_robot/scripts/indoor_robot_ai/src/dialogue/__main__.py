"""CLI: interactive or one-shot dialogue.

Examples:

  PYTHONPATH= python -m src.dialogue "Bring me the bottle."
  PYTHONPATH= python -m src.dialogue --interactive
"""

from __future__ import annotations

import argparse
import json
import sys

from src.dialogue.dialogue_manager import DialogueManager
from src.dialogue.types import DialogueStatus
from src.logging_utils import setup_logging
from src.nlu.nlu import NLUEngine


def _print_result(result, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
        return
    print(f"status: {result.status.value}")
    if result.question:
        print(f"robot: {result.question}")
    elif result.message:
        print(f"robot: {result.message}")
    if result.task:
        print(f"task: {json.dumps(result.task, indent=2)}")


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Phase 5 dialogue manager. No planning or robot control."
    )
    parser.add_argument("text", nargs="*", help="One-shot user utterance.")
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Multi-turn conversation (type 'quit' to exit).",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON results.")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Train Phase 3 intent model from data/nlu/train.jsonl.",
    )
    args = parser.parse_args(argv)

    engine = NLUEngine.train_default() if args.retrain else NLUEngine.load()
    manager = DialogueManager(engine=engine)

    if args.interactive or not args.text:
        if not args.interactive and not args.text:
            # default to interactive when no text
            pass
        if args.text and not args.interactive:
            result = manager.process(" ".join(args.text))
            _print_result(result, args.json)
            return 0

        print("Dialogue mode. Type 'quit' to exit. Type 'reset' to clear state.")
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not line:
                continue
            if line.lower() in {"quit", "exit"}:
                return 0
            if line.lower() == "reset":
                manager.reset()
                print("state cleared")
                continue
            result = manager.process(line)
            if args.json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                if result.status == DialogueStatus.CLARIFICATION_REQUIRED:
                    print(f"robot> {result.question}")
                elif result.status == DialogueStatus.READY:
                    print(f"robot> Ready.")
                    print(f"task> {json.dumps(result.task)}")
                elif result.status == DialogueStatus.CANCELLED:
                    print("robot> Cancelled.")
                elif result.status == DialogueStatus.UNSUPPORTED:
                    print("robot> Unsupported.")
                elif result.status == DialogueStatus.CLARIFICATION_FAILED:
                    print("robot> Clarification failed.")
                else:
                    print(f"robot> {result.status.value}")
        return 0

    result = manager.process(" ".join(args.text))
    _print_result(result, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
