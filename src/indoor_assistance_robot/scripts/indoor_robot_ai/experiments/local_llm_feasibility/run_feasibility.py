#!/usr/bin/env python3
"""Run the isolated Phase 9A local LLM feasibility diagnostic.

This script evaluates language interpretation only. It does not execute robot
plans and does not call remote inference services.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from llama_cpp import Llama


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def meminfo() -> Dict[str, float]:
    values: Dict[str, float] = {}
    with Path("/proc/meminfo").open(encoding="utf-8") as handle:
        for line in handle:
            key, raw = line.split(":", 1)
            parts = raw.strip().split()
            if parts and parts[0].isdigit():
                values[key] = int(parts[0]) / 1024.0
    swap_total = values.get("SwapTotal", 0.0)
    swap_free = values.get("SwapFree", 0.0)
    values["SwapUsed"] = swap_total - swap_free
    return values


def process_memory_mib() -> Dict[str, float]:
    values: Dict[str, float] = {}
    with Path("/proc/self/status").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("VmRSS:", "VmSize:", "VmHWM:")):
                key, raw = line.split(":", 1)
                parts = raw.strip().split()
                if parts and parts[0].isdigit():
                    values[key] = int(parts[0]) / 1024.0
    return values


def memory_snapshot() -> Dict[str, Any]:
    info = meminfo()
    return {
        "system_mib": {
            "mem_available": info.get("MemAvailable"),
            "mem_free": info.get("MemFree"),
            "swap_total": info.get("SwapTotal"),
            "swap_free": info.get("SwapFree"),
            "swap_used": info.get("SwapUsed"),
        },
        "process_mib": process_memory_mib(),
    }


def should_stop_for_memory(
    snapshot: Dict[str, Any],
    *,
    min_available_mib: float,
    initial_swap_free_mib: float,
    max_swap_drop_mib: float,
) -> Optional[str]:
    system = snapshot["system_mib"]
    available = float(system.get("mem_available") or 0.0)
    swap_free = float(system.get("swap_free") or 0.0)
    if available < min_available_mib:
        return (
            f"MemAvailable {available:.1f} MiB is below "
            f"{min_available_mib:.1f} MiB."
        )
    if initial_swap_free_mib - swap_free > max_swap_drop_mib:
        return (
            f"SwapFree dropped by {initial_swap_free_mib - swap_free:.1f} MiB, "
            f"above limit {max_swap_drop_mib:.1f} MiB."
        )
    return None


def extract_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not text or not text.strip():
        return None, "empty output"
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    last_error: Optional[str] = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            continue
        if isinstance(parsed, dict):
            return parsed, None
        last_error = "JSON root is not an object"
    return None, last_error or "no JSON object found"


def validate_type(value: Any, spec: Any, path: str, errors: List[str]) -> None:
    expected = spec.get("type")
    if isinstance(expected, list):
        if "null" in expected and value is None:
            return
        non_null = [item for item in expected if item != "null"]
        for item in non_null:
            if type_matches(value, item):
                return
        errors.append(f"{path} expected one of {expected}, got {type(value).__name__}")
        return
    if expected and not type_matches(value, expected):
        errors.append(f"{path} expected {expected}, got {type(value).__name__}")
        return
    if "allowed" in spec and value not in spec["allowed"]:
        errors.append(f"{path} has unsupported value {value!r}")
    if expected == "object":
        required = spec.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{path}.{key} is missing")
        props = spec.get("properties", {})
        for key, child in props.items():
            if key in value:
                validate_type(value[key], child, f"{path}.{key}", errors)
    if expected == "array":
        if "items" in spec:
            for index, item in enumerate(value):
                validate_type(item, spec["items"], f"{path}[{index}]", errors)


def type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def validate_output(payload: Optional[Dict[str, Any]], schema: Dict[str, Any]) -> List[str]:
    if payload is None:
        return ["no parsed JSON payload"]
    errors: List[str] = []
    validate_type(payload, schema, "$", errors)
    return errors


def prompt_for_case(case: Dict[str, Any], config: Dict[str, Any]) -> str:
    catalogs = config["catalogs"]
    schema_summary = {
        "intent": "NAVIGATE|RETRIEVE|DELIVER|FOLLOW|STOP|RETURN|UNSUPPORTED|UNKNOWN",
        "entities": {
            "object": "canonical object id or null",
            "source_location": "canonical location id or null",
            "destination": "canonical location/user id or null",
            "person": "canonical person id or null",
        },
        "missing_information": ["slot names or short descriptions"],
        "ambiguity": [
            {"field": "slot", "reason": "why ambiguous", "candidates": ["ids"]}
        ],
        "unsupported_task_reasons": ["reason strings"],
        "corrections": [
            {"target": "field/task", "replacement": "new value", "reason": "why"}
        ],
        "contextual_references": [
            {"text": "it/here/there/instead/etc", "resolved_to": None, "status": "resolved|unresolved"}
        ],
        "multi_step_task_coverage": {
            "has_multi_step_request": False,
            "steps": [
                {
                    "description": "natural-language subtask",
                    "support_status": "SUPPORTED|UNSUPPORTED|UNKNOWN",
                    "required_entities": ["slots"],
                    "missing_information": ["missing slots"],
                    "ambiguity": ["ambiguous parts"],
                }
            ],
        },
    }
    return (
        "You are evaluating language understanding for an indoor assistance robot.\n"
        "Return ONLY one JSON object. No markdown. No prose.\n"
        "Do NOT produce executable plans, ROS commands, /cmd_vel, joint commands, "
        "coordinates, scripts, or motor-control instructions.\n"
        "You may return UNKNOWN or UNSUPPORTED. Do not force a request into a "
        "supported intent.\n"
        "Supported intents: NAVIGATE, RETRIEVE, DELIVER, FOLLOW, STOP, RETURN.\n"
        f"Allowed objects: {catalogs['objects']}\n"
        f"Allowed locations: {catalogs['locations']}\n"
        f"Allowed persons: {catalogs['persons']}\n"
        f"Output shape: {json.dumps(schema_summary, sort_keys=True)}\n"
        f"Context: {case.get('context')}\n"
        f"User command: {case['text']}\n"
    )


def run_baseline(case: Dict[str, Any]) -> Dict[str, Any]:
    model_path = PROJECT_ROOT / "models" / "nlu" / "intent_tfidf_lr.joblib"
    if not model_path.is_file():
        return {"available": False, "error": f"saved NLU model not found: {model_path}"}
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from src.nlu.nlu import NLUEngine

        engine = NLUEngine.load(model_path)
        result = engine.parse(case["text"])
        return {
            "available": True,
            "intent": result.intent,
            "intent_confidence": result.intent_confidence,
            "entities": dict(result.entities),
        }
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {
            "available": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def run(args: argparse.Namespace) -> int:
    config = read_json(Path(args.config))
    schema = read_json(Path(args.schema))
    cases = load_jsonl(Path(args.cases))
    model_path = EXPERIMENT_DIR / config["model"]["local_path"]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    runtime = config["runtime"]
    report: Dict[str, Any] = {
        "experiment": config["experiment_name"],
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "model": config["model"],
        "runtime": runtime,
        "notes": [
            "Process memory is from /proc/self/status for this Python process.",
            "System memory is from /proc/meminfo for the whole operating system.",
            "This is a 10-case diagnostic, not a generalization benchmark.",
        ],
        "environment": {
            "python": sys.version,
            "pid": os.getpid(),
        },
        "memory_before_load": memory_snapshot(),
        "load": {},
        "cases": [],
        "summary": {},
    }

    initial_swap_free = float(
        report["memory_before_load"]["system_mib"].get("swap_free") or 0.0
    )
    stop_reason = should_stop_for_memory(
        report["memory_before_load"],
        min_available_mib=float(runtime["memory_stop_available_mib"]),
        initial_swap_free_mib=initial_swap_free,
        max_swap_drop_mib=float(runtime["swap_drop_stop_mib"]),
    )
    if stop_reason:
        report["load"] = {"success": False, "error": stop_reason}
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 2

    started = time.perf_counter()
    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=int(runtime["n_ctx"]),
            n_threads=int(runtime["n_threads"]),
            n_gpu_layers=int(runtime["n_gpu_layers"]),
            verbose=False,
        )
    except Exception as exc:
        report["load"] = {
            "success": False,
            "load_time_seconds": time.perf_counter() - started,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        report["memory_after_load_attempt"] = memory_snapshot()
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 1

    report["load"] = {
        "success": True,
        "load_time_seconds": time.perf_counter() - started,
    }
    report["memory_after_load"] = memory_snapshot()
    stop_reason = should_stop_for_memory(
        report["memory_after_load"],
        min_available_mib=float(runtime["memory_stop_available_mib"]),
        initial_swap_free_mib=initial_swap_free,
        max_swap_drop_mib=float(runtime["swap_drop_stop_mib"]),
    )
    if stop_reason:
        report["stopped_before_inference"] = stop_reason
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 3

    for case in cases:
        before = memory_snapshot()
        stop_reason = should_stop_for_memory(
            before,
            min_available_mib=float(runtime["memory_stop_available_mib"]),
            initial_swap_free_mib=initial_swap_free,
            max_swap_drop_mib=float(runtime["swap_drop_stop_mib"]),
        )
        if stop_reason:
            report["cases"].append(
                {
                    "id": case["id"],
                    "text": case["text"],
                    "skipped": True,
                    "skip_reason": stop_reason,
                }
            )
            break

        prompt = prompt_for_case(case, config)
        infer_started = time.perf_counter()
        raw_text = ""
        error = None
        try:
            response = llm(
                prompt,
                max_tokens=int(runtime["max_tokens"]),
                temperature=float(runtime["temperature"]),
                top_p=float(runtime["top_p"]),
                repeat_penalty=float(runtime["repeat_penalty"]),
                stop=["\n\nUser command:", "\n\nContext:"],
            )
            raw_text = str(response["choices"][0]["text"])
        except Exception as exc:
            error = str(exc)
        latency = time.perf_counter() - infer_started
        parsed, parse_error = extract_json_object(raw_text)
        validation_errors = validate_output(parsed, schema)
        baseline = run_baseline(case) if args.baseline else {"available": False}
        report["cases"].append(
            {
                "id": case["id"],
                "text": case["text"],
                "context": case.get("context"),
                "notes": case.get("notes"),
                "latency_seconds": latency,
                "memory_before": before,
                "memory_after": memory_snapshot(),
                "raw_output": raw_text,
                "inference_error": error,
                "parsed_output": parsed,
                "parse_error": parse_error,
                "schema_errors": validation_errors,
                "structured_output_valid": parsed is not None and not validation_errors,
                "baseline_nlu": baseline,
            }
        )

    valid_count = sum(1 for item in report["cases"] if item.get("structured_output_valid"))
    report["summary"] = {
        "n_cases_requested": len(cases),
        "n_cases_run": len([item for item in report["cases"] if not item.get("skipped")]),
        "n_structured_valid": valid_count,
        "structured_valid_rate": valid_count / len(report["cases"]) if report["cases"] else 0.0,
        "completed_all_cases": len(report["cases"]) == len(cases)
        and not any(item.get("skipped") for item in report["cases"]),
    }
    report["memory_after_all"] = memory_snapshot()
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["summary"]["completed_all_cases"] else 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(EXPERIMENT_DIR / "config.json"),
    )
    parser.add_argument(
        "--schema",
        default=str(EXPERIMENT_DIR / "schema.json"),
    )
    parser.add_argument(
        "--cases",
        default=str(EXPERIMENT_DIR / "cases.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(EXPERIMENT_DIR / "results" / "diagnostic_report.json"),
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Compare with the existing saved NLU model using NLUEngine.load only.",
    )
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
