"""CLI: python -m src.validation "Bring me the bottle." """

from __future__ import annotations

import argparse
import json

from src.logging_utils import setup_logging
from src.nlu.nlu import NLUEngine, parse
from src.validation.validator import validate


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Validate a typed command. Does not control the robot."
    )
    parser.add_argument("text", nargs="+", help="Command text (quote the sentence).")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Train the Phase 3 intent model from data/nlu/train.jsonl.",
    )
    args = parser.parse_args(argv)
    text = " ".join(args.text)
    engine = NLUEngine.train_default() if args.retrain else NLUEngine.load()
    result = validate(parse(text, engine=engine))
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
