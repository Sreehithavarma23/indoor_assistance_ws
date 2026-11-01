"""Dry-run simulation adapter.

Logs recognized command, entities, validation, structured plan, intended
operation, and rejection reason. Does **not** publish ROS commands unless
explicitly opted in later after interface verification.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.config_loader import (
    PROJECT_ROOT,
    load_locations,
    load_objects,
    load_persons,
)
from src.logging_utils import get_logger
from src.pipeline.orchestrator import AIPipeline
from src.pipeline.types import PipelineResult, PipelineStatus
from src.planning.exceptions import PlanValidationError
from src.planning.plan_validator import validate_plan
from src.planning.types import ActionType, PlanningStatus

logger = get_logger("simulation.dry_run")

CAPABILITY_MAP = PROJECT_ROOT / "src" / "simulation" / "capability_map.yaml"
DEFAULT_LOG_DIR = PROJECT_ROOT / "results" / "simulation"


@dataclass
class DryRunResult:
    timestamp: str
    input_text: str
    transcript: Optional[str]
    intent: Optional[str]
    entities: Dict[str, str]
    dialogue_status: Optional[str]
    pipeline_status: str
    validation_ok: bool
    plan_actions: List[Dict[str, Any]]
    intended_operations: List[Dict[str, Any]]
    rejection_reason: Optional[str]
    would_execute: bool
    adapter_status: str = "REJECTED"
    success: bool = False
    capability_notes: List[str] = field(default_factory=list)
    raw_pipeline: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_capability_map(path: Optional[Path] = None) -> dict:
    target = path or CAPABILITY_MAP
    with target.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class DryRunAdapter:
    """AI pipeline → logged intended ops. Never sends robot commands."""

    def __init__(
        self,
        *,
        pipeline: Optional[AIPipeline] = None,
        capability_map: Optional[dict] = None,
        allow_live_ros: bool = False,
        log_dir: Optional[Path] = None,
    ) -> None:
        if allow_live_ros:
            raise ValueError(
                "Live ROS execution is disabled in this semester build. "
                "Keep allow_live_ros=False and use dry-run logging only."
            )
        self.pipeline = pipeline or AIPipeline()
        self.capability_map = capability_map or load_capability_map()
        self.log_dir = Path(log_dir or DEFAULT_LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.allow_live_ros = False

    def _bind_actions(self, plan_actions: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[str], Optional[str]]:
        bindings = self.capability_map.get("action_bindings") or {}
        intended: List[Dict[str, Any]] = []
        notes: List[str] = []
        rejection: Optional[str] = None
        for action in plan_actions:
            atype = str(action.get("type") or action.get("action_type") or "")
            spec = bindings.get(atype) or {}
            status = spec.get("status", "unknown")
            entry = {
                "action_type": atype,
                "parameters": action.get("parameters") or {},
                "binding_status": status,
                "intended_interface": spec.get("intended_interface"),
                "available_today": bool(spec.get("available_today", False)),
                "reason": spec.get("reason") or spec.get("note"),
            }
            intended.append(entry)
            if status != "dry_run_only" or not entry["available_today"]:
                notes.append(
                    f"{atype}: {status} — {entry.get('reason') or 'not executable in current sim'}"
                )
        if intended and not any(item.get("available_today") for item in intended):
            rejection = (
                "No verified executable binding for household actions in the "
                "surgical simulator; dry-run only."
            )
        return intended, notes, rejection

    @staticmethod
    def _validate_plan(result: PipelineResult) -> Optional[str]:
        """Validate the plan again at the adapter boundary before logging ops."""
        if result.status != PipelineStatus.PLANNED or result.plan is None:
            return "Pipeline did not produce a validated plan."
        plan = result.plan
        if plan.status != PlanningStatus.PLANNED:
            return f"Plan status {plan.status.value} is not executable as a dry run."
        try:
            validate_plan(plan)
        except PlanValidationError as exc:
            return f"Malformed plan: {exc}"

        location_ids = {
            str(item["id"]) for item in load_locations().get("locations", [])
        }
        object_ids = {str(item["id"]) for item in load_objects().get("objects", [])}
        person_ids = {str(item["id"]) for item in load_persons().get("persons", [])}
        for action in plan.actions:
            params = action.parameters
            if action.type == ActionType.NAVIGATE_TO:
                if params.get("destination") not in location_ids:
                    return f"Unknown destination {params.get('destination')!r}."
            elif action.type == ActionType.SEARCH_OBJECT:
                if params.get("location") not in location_ids:
                    return f"Unknown source location {params.get('location')!r}."
                if params.get("object") not in object_ids:
                    return f"Unknown object {params.get('object')!r}."
            elif action.type in {ActionType.APPROACH_OBJECT, ActionType.GRASP_OBJECT}:
                if params.get("object") not in object_ids:
                    return f"Unknown object {params.get('object')!r}."
            elif action.type == ActionType.DELIVER_OBJECT:
                if params.get("object") not in object_ids:
                    return f"Unknown object {params.get('object')!r}."
                destination = params.get("destination")
                if destination and destination != "user" and destination not in location_ids:
                    return f"Unknown destination {destination!r}."
                person = params.get("person")
                if person and person not in person_ids:
                    return f"Unknown person {person!r}."
            elif action.type in {ActionType.FOLLOW_PERSON, ActionType.NAVIGATE_TO_PERSON}:
                if params.get("person") not in person_ids:
                    return f"Unknown person {params.get('person')!r}."
        return None

    def run_text(self, text: str, *, mode: str = "deterministic") -> DryRunResult:
        result: PipelineResult = self.pipeline.run_text(text, mode=mode)
        return self._from_pipeline_result(text, result)

    def run_audio(self, audio_path: str | Path, *, mode: str = "deterministic") -> DryRunResult:
        """Run the file-based ASR pipeline and keep the same dry-run gate."""
        result: PipelineResult = self.pipeline.run_audio(audio_path, mode=mode)
        return self._from_pipeline_result(str(audio_path), result)

    def _from_pipeline_result(
        self,
        fallback_input: str,
        result: PipelineResult,
    ) -> DryRunResult:
        plan_actions: List[Dict[str, Any]] = []
        if result.plan is not None:
            for action in result.plan.actions:
                plan_actions.append(
                    {
                        "type": getattr(action.type, "value", str(action.type)),
                        "parameters": dict(action.parameters),
                        "id": getattr(action, "id", None),
                    }
                )

        rejection = self._validate_plan(result)
        intended: List[Dict[str, Any]] = []
        notes: List[str] = []
        binding_rejection: Optional[str] = None
        if rejection is None:
            intended, notes, binding_rejection = self._bind_actions(plan_actions)
        would_execute = False
        validation_ok = rejection is None

        if rejection is None and binding_rejection:
            rejection = binding_rejection
        elif rejection is None:
            # Still never execute in this build.
            rejection = "Dry-run mode: execution gated off (allow_live_ros=False)."
            would_execute = False

        adapter_status = "DRY_RUN_ACCEPTED" if validation_ok else "REJECTED"

        dry = DryRunResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_text=result.input_text or fallback_input,
            transcript=result.transcript.text if result.transcript else None,
            intent=result.intent,
            entities=dict(result.entities or {}),
            dialogue_status=result.dialogue_status,
            pipeline_status=result.status.value,
            validation_ok=validation_ok,
            plan_actions=plan_actions,
            intended_operations=intended,
            rejection_reason=rejection,
            would_execute=would_execute,
            adapter_status=adapter_status,
            success=validation_ok,
            capability_notes=notes,
            raw_pipeline=result.to_dict() if hasattr(result, "to_dict") else {},
        )
        self._append_log(dry)
        logger.info(
            "Dry-run intent=%s status=%s would_execute=%s reason=%s",
            dry.intent,
            dry.pipeline_status,
            dry.would_execute,
            dry.rejection_reason,
        )
        return dry

    def _append_log(self, dry: DryRunResult) -> Path:
        path = self.log_dir / "dry_run.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dry.to_dict(), ensure_ascii=False) + "\n")
        return path


def main(argv: list[str] | None = None) -> int:
    import argparse

    from src.logging_utils import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser(description="Dry-run speech/text → plan adapter.")
    parser.add_argument("text", nargs="?", default="Take me to the kitchen.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    adapter = DryRunAdapter()
    result = adapter.run_text(args.text)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"intent={result.intent} status={result.pipeline_status}")
        print(f"entities={result.entities}")
        print(f"plan={result.plan_actions}")
        print(f"rejection={result.rejection_reason}")
        print(f"would_execute={result.would_execute}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
