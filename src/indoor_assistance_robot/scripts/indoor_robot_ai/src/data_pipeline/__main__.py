"""Build the NLU v2 corpus, splits, BIO export, and dataset report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config_loader import PROJECT_ROOT
from src.data_pipeline.bio import BIO_DIR, export_bio_jsonl
from src.data_pipeline.generate import generate_all, write_raw_corpus
from src.data_pipeline.report import build_report, write_markdown_report
from src.data_pipeline.split import check_leakage, split_records, write_splits
from src.logging_utils import setup_logging


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Build leakage-aware NLU v2 dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-family", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "nlu_v2"),
    )
    args = parser.parse_args(argv)

    records = generate_all(seed=args.seed, max_per_family=args.max_per_family)
    raw_path = write_raw_corpus(
        records, path=Path(args.output_dir) / "raw" / "corpus.jsonl"
    )
    splits = split_records(records, seed=args.seed)
    issues = check_leakage(splits)
    if issues:
        # Soft fail: still write, but surface issues in report and stderr.
        print(f"WARNING: {len(issues)} leakage issue(s) detected.")
        for issue in issues[:20]:
            print(f"  - {issue}")
    splits_dir = write_splits(splits, directory=Path(args.output_dir) / "splits")
    for name, rows in splits.items():
        export_bio_jsonl(rows, BIO_DIR / f"{name}.jsonl")
    report = build_report(splits, leakage_issues=issues)
    report_path = write_markdown_report(
        report, path=Path(args.output_dir) / "DATASET_REPORT.md"
    )
    summary = {
        "raw_corpus": str(raw_path),
        "splits_dir": str(splits_dir),
        "report": str(report_path),
        "n_total": report["n_total"],
        "per_split": report["per_split"],
        "n_leakage_issues": len(issues),
    }
    print(json.dumps(summary, indent=2))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
