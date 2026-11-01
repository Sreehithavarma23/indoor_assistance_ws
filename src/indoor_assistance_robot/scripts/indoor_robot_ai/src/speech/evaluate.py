"""Evaluate ASR predictions against a reference manifest.

Manifest CSV columns: audio_path,reference
Lines starting with # are ignored.

This script does not invent scores. An empty or missing recording set
produces n_items=0 and no fabricated averages.
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
from pathlib import Path
from typing import Any, Dict, List

import soundfile as sf

from src.config_loader import PROJECT_ROOT, load_speech
from src.config_loader import load_locations, load_objects, load_persons
from src.logging_utils import get_logger, setup_logging
from src.speech.asr import WhisperASR, resolve_compute_type, resolve_device
from src.speech.metrics import character_error_rate, normalize_text, word_error_rate
from src.speech.pipeline import transcribe_file

logger = get_logger("speech.evaluate")


def _catalog_terms() -> List[str]:
    terms: List[str] = []
    for item in load_objects().get("objects") or []:
        terms.append(str(item.get("id", "")).replace("_", " "))
        terms.extend(str(alias) for alias in item.get("aliases") or [])
    for item in load_locations().get("locations") or []:
        terms.append(str(item.get("id", "")).replace("_", " "))
        terms.extend(str(alias) for alias in item.get("aliases") or [])
    for item in load_persons().get("persons") or []:
        terms.append(str(item.get("id", "")).replace("_", " "))
        terms.extend(str(alias) for alias in item.get("aliases") or [])
    terms.extend(["not", "don't", "do not", "actually", "instead", "other"])
    return [term for term in terms if term.strip()]


def _critical_terms_for_row(row: Dict[str, str], reference_norm: str) -> List[str]:
    configured = (row.get("critical_terms") or "").strip()
    if configured:
        raw_terms = configured.replace("|", ";").split(";")
        return [term.strip() for term in raw_terms if term.strip()]
    terms = []
    for term in _catalog_terms():
        normalized = normalize_text(term)
        if normalized and normalized in reference_norm:
            terms.append(term)
    return sorted(set(terms))


def critical_term_error_rate(
    reference: str,
    hypothesis: str,
    critical_terms: List[str],
) -> float | None:
    """Fraction of expected critical terms absent from the hypothesis."""
    normalized_terms = [normalize_text(term) for term in critical_terms]
    normalized_terms = [term for term in normalized_terms if term]
    if not normalized_terms:
        return None
    misses = sum(1 for term in normalized_terms if term not in hypothesis)
    return misses / len(normalized_terms)


def _audio_duration_seconds(path: Path) -> float | None:
    try:
        info = sf.info(path)
    except Exception:
        return None
    if not info.samplerate:
        return None
    return float(info.frames) / float(info.samplerate)


def _memory_mb() -> float:
    # Linux ru_maxrss is KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def read_manifest(path: str | Path) -> List[Dict[str, str]]:
    """Load evaluation pairs. Missing files are reported later, not invented."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows: List[Dict[str, str]] = []
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        filtered = [
            line
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]
        reader = csv.DictReader(filtered)
        if reader.fieldnames is None:
            return []
        required = {"audio_path", "reference"}
        missing = required.difference({name.strip() for name in reader.fieldnames})
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        for row in reader:
            audio_path = (row.get("audio_path") or "").strip()
            reference = (row.get("reference") or "").strip()
            if not audio_path:
                continue
            rows.append(
                {
                    "audio_path": audio_path,
                    "reference": reference,
                    "critical_terms": (row.get("critical_terms") or "").strip(),
                }
            )
    return rows


def evaluate_manifest(manifest_path: str | Path) -> Dict[str, Any]:
    """Run the speech pipeline on each row and compute WER/CER/latency."""
    setup_logging()
    config = load_speech()
    eval_settings = config.get("evaluation", {})
    asr_settings = config.get("asr", {})

    items: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    asr = WhisperASR()
    device = resolve_device(str(asr_settings.get("device", "auto")))
    compute_type = resolve_compute_type(
        str(asr_settings.get("compute_type", "default")),
        device,
    )
    memory_before_mb = _memory_mb()

    for row in read_manifest(manifest_path):
        audio_path = Path(row["audio_path"])
        if not audio_path.is_absolute():
            audio_path = PROJECT_ROOT / audio_path
        if not audio_path.is_file():
            logger.warning("Skipping missing audio file: %s", audio_path)
            skipped.append(
                {
                    "audio_path": str(audio_path),
                    "reason": "file_not_found",
                    "reference": row["reference"],
                }
            )
            continue

        duration = _audio_duration_seconds(audio_path)
        transcript = transcribe_file(audio_path, asr=asr)
        reference_norm = normalize_text(
            row["reference"],
            lowercase=bool(eval_settings.get("lowercase", True)),
            remove_punctuation=bool(eval_settings.get("remove_punctuation", True)),
        )
        prediction_norm = normalize_text(
            transcript.text,
            lowercase=bool(eval_settings.get("lowercase", True)),
            remove_punctuation=bool(eval_settings.get("remove_punctuation", True)),
        )
        critical_terms = _critical_terms_for_row(row, reference_norm)
        pipeline_latency = transcript.pipeline_latency_seconds
        rtf = None
        if duration and pipeline_latency is not None:
            rtf = pipeline_latency / duration
        items.append(
            {
                "audio_path": str(audio_path),
                "reference": row["reference"],
                "prediction": transcript.text,
                "critical_terms": critical_terms,
                "critical_term_error_rate": critical_term_error_rate(
                    reference_norm,
                    prediction_norm,
                    critical_terms,
                ),
                "wer": word_error_rate(reference_norm, prediction_norm),
                "cer": character_error_rate(reference_norm, prediction_norm),
                "audio_duration_seconds": duration,
                "latency_seconds": transcript.latency_seconds,
                "pipeline_latency_seconds": transcript.pipeline_latency_seconds,
                "real_time_factor": rtf,
                "speech_detected": transcript.speech_detected,
            }
        )
    memory_after_mb = _memory_mb()

    summary: Dict[str, Any] = {
        "backend": "faster_whisper",
        "model_size": asr_settings.get("model_size"),
        "device": device,
        "compute_type": compute_type,
        "language": asr_settings.get("language"),
        "n_items": len(items),
        "n_skipped": len(skipped),
        "mean_wer": None,
        "mean_cer": None,
        "mean_critical_term_error_rate": None,
        "mean_latency_seconds": None,
        "mean_pipeline_latency_seconds": None,
        "mean_real_time_factor": None,
        "model_load_seconds": asr.model_load_seconds,
        "memory_before_mb": memory_before_mb,
        "memory_after_mb": memory_after_mb,
        "memory_peak_delta_mb": max(0.0, memory_after_mb - memory_before_mb),
        "note": (
            "Real WER/CER require actual speech recordings and reference text. "
            "This file reports only items that were actually transcribed."
        ),
        "items": items,
        "skipped": skipped,
    }
    if items:
        summary["mean_wer"] = sum(item["wer"] for item in items) / len(items)
        summary["mean_cer"] = sum(item["cer"] for item in items) / len(items)
        critical_scores = [
            item["critical_term_error_rate"]
            for item in items
            if item["critical_term_error_rate"] is not None
        ]
        if critical_scores:
            summary["mean_critical_term_error_rate"] = (
                sum(critical_scores) / len(critical_scores)
            )
        summary["mean_latency_seconds"] = sum(
            item["latency_seconds"] for item in items
        ) / len(items)
        summary["mean_pipeline_latency_seconds"] = sum(
            item["pipeline_latency_seconds"] for item in items
        ) / len(items)
        rtfs = [
            item["real_time_factor"]
            for item in items
            if item["real_time_factor"] is not None
        ]
        if rtfs:
            summary["mean_real_time_factor"] = sum(rtfs) / len(rtfs)
    return summary


def write_report(summary: Dict[str, Any], output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote evaluation report to %s", destination)
    return destination


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Faster-Whisper transcripts against a reference manifest."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="CSV with columns audio_path,reference",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "asr" / "eval.json"),
        help="JSON report path",
    )
    args = parser.parse_args(argv)

    summary = evaluate_manifest(args.manifest)
    write_report(summary, args.output)
    print(json.dumps({k: summary[k] for k in summary if k not in {"items", "skipped"}}, indent=2))
    if summary["n_items"] == 0:
        print(
            "No audio files were evaluated. Add recordings to the manifest; "
            "scores are not fabricated."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
