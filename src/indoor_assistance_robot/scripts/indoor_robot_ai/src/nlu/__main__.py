"""CLI: python -m src.nlu "Can you take me to the kitchen?" """

from __future__ import annotations

import argparse
import json

from src.logging_utils import setup_logging
from src.nlu.nlu import NLUEngine, parse


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Parse a typed command with the Phase 3 NLU baseline."
    )
    parser.add_argument("text", nargs="+", help="Command text (quote the sentence).")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Train from data/nlu/train.jsonl instead of loading a saved model.",
    )
    args = parser.parse_args(argv)
    text = " ".join(args.text)
    engine = NLUEngine.train_default() if args.retrain else NLUEngine.load()
    result = parse(text, engine=engine)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
