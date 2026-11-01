"""Dataset report: counts, provenance, leakage, limitations."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.config_loader import PROJECT_ROOT
from src.data_pipeline.schema import DatasetRecord
from src.data_pipeline.split import check_leakage
from src.nlu.normalization import normalize_text

REPORT_PATH = PROJECT_ROOT / "data" / "nlu_v2" / "DATASET_REPORT.md"


def _entity_counts(records: Sequence[DatasetRecord]) -> Dict[str, int]:
    counts: Counter = Counter()
    for rec in records:
        for slot in rec.entities:
            counts[slot] += 1
    return dict(counts)


def build_report(
    splits: Dict[str, List[DatasetRecord]],
    leakage_issues: Sequence[str] | None = None,
) -> Dict[str, Any]:
    all_rows = [r for rows in splits.values() for r in rows]
    leakage = list(leakage_issues) if leakage_issues is not None else check_leakage(splits)

    # Near-duplicate heuristic: same first 4 tokens across different splits.
    def prefix(text: str) -> str:
        toks = normalize_text(text).split()[:4]
        return " ".join(toks)

    prefix_map: Dict[str, set] = {}
    for name, rows in splits.items():
        for rec in rows:
            prefix_map.setdefault(prefix(rec.text), set()).add(name)
    near = sum(1 for splits_present in prefix_map.values() if len(splits_present) > 1)

    return {
        "n_total": len(all_rows),
        "per_split": {name: len(rows) for name, rows in splits.items()},
        "intent_counts_global": dict(Counter(r.intent for r in all_rows)),
        "intent_counts_per_split": {
            name: dict(Counter(r.intent for r in rows)) for name, rows in splits.items()
        },
        "source_counts": dict(Counter(r.source for r in all_rows)),
        "entity_slot_counts": _entity_counts(all_rows),
        "n_with_task_order": sum(1 for r in all_rows if r.task_order),
        "n_families": len({r.family_id for r in all_rows}),
        "leakage_issues": list(leakage),
        "near_duplicate_prefix_cross_split": near,
        "limitations": [
            "Most utterances are synthetic_template or synthetic_paraphrase, not natural speech.",
            "ASR noise is approximate character/word corruption, not real Whisper errors.",
            "public_aux_oos examples are hand-authored methodology references, not redistributed CLINC/SNIPS/MASSIVE rows.",
            "Frozen test sets are project-internal; they are not an external independent benchmark.",
            "No consented multi-speaker audio corpus is bundled; ASR WER cannot be claimed from this release.",
            "Multi-step commands use a single primary intent label for classifier training; task_order is auxiliary.",
        ],
    }


def write_markdown_report(report: Dict[str, Any], path: Path | None = None) -> Path:
    out = path or REPORT_PATH
    lines = [
        "# NLU v2 Dataset Report",
        "",
        f"**Total utterances:** {report['n_total']}",
        f"**Families:** {report['n_families']}",
        "",
        "## Split sizes",
        "",
        "| Split | Count |",
        "|---|---:|",
    ]
    for name, n in report["per_split"].items():
        lines.append(f"| `{name}` | {n} |")
    lines.extend(
        [
            "",
            "## Intent distribution (global)",
            "",
            "| Intent | Count |",
            "|---|---:|",
        ]
    )
    for intent, n in sorted(report["intent_counts_global"].items()):
        lines.append(f"| {intent} | {n} |")
    lines.extend(
        [
            "",
            "## Provenance (source)",
            "",
            "| Source | Count |",
            "|---|---:|",
        ]
    )
    for source, n in sorted(report["source_counts"].items()):
        lines.append(f"| `{source}` | {n} |")
    lines.extend(
        [
            "",
            "## Entity slot mentions",
            "",
            "| Slot | Count |",
            "|---|---:|",
        ]
    )
    for slot, n in sorted(report["entity_slot_counts"].items()):
        lines.append(f"| `{slot}` | {n} |")
    lines.extend(
        [
            "",
            f"**Multi-step (`task_order`) rows:** {report['n_with_task_order']}",
            f"**Near-duplicate prefixes spanning splits:** {report['near_duplicate_prefix_cross_split']}",
            "",
            "## Leakage checks",
            "",
        ]
    )
    if report["leakage_issues"]:
        for issue in report["leakage_issues"]:
            lines.append(f"- {issue}")
    else:
        lines.append("- No exact-normalized or train↔frozen-test family overlaps detected.")
    lines.extend(["", "## Known limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    (out.parent / "dataset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out
