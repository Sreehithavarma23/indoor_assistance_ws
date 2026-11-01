"""Build grounded prompts for constrained LLM planning."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from src.planning.action_registry import ACTION_REGISTRY
from src.validation.schema import load_policy


def _action_schema() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for action_type, spec in ACTION_REGISTRY.items():
        rows.append(
            {
                "type": action_type.value,
                "required_parameters": list(spec.required_parameters),
                "optional_parameters": list(spec.optional_parameters),
                "purpose": spec.purpose,
            }
        )
    return rows


def build_planning_prompt(
    *,
    intent: str,
    entities: Dict[str, str],
    original_text: str,
    correction_errors: Sequence[str] | None = None,
) -> str:
    """Assemble a single user prompt with catalogs, schema, and task context."""
    policy = load_policy()
    schema = {
        "status": "PLANNED",
        "intent": "<same intent or UNSUPPORTED>",
        "entities": {"<slot>": "<canonical_id>"},
        "actions": [
            {
                "id": "action_1",
                "type": "<ActionType>",
                "parameters": {"<param>": "<canonical_id>"},
                "order": 1,
                "depends_on": [],
                "description": "<short>",
            }
        ],
        "explanation": "<short>",
        "errors": [],
        "warnings": [],
    }
    rules = [
        "Return ONLY a single JSON object. No markdown fences. No commentary.",
        "Use ONLY action types from the allowed action vocabulary.",
        "Use ONLY canonical object/location/person IDs from the catalogs.",
        "Parameter names must match the action schema exactly "
        "(e.g. NAVIGATE_TO uses 'destination', not 'location').",
        "Action ids must be unique. depends_on may only reference existing ids.",
        "Dependencies must precede the dependent action (smaller order).",
        "Do not invent capabilities, coordinates, joint angles, or ROS topics.",
        "Do not assume an object is held unless the task already implies "
        "DELIVER without a source (document a warning if so).",
        "If the task cannot be planned safely, return status FAILED or "
        "UNSUPPORTED with empty actions and errors explaining why.",
        "Ground the plan in the provided structured intent and entities; "
        "the original text is context only.",
    ]
    payload = {
        "rules": rules,
        "allowed_actions": _action_schema(),
        "catalogs": {
            "objects": sorted(policy.object_ids),
            "locations": sorted(policy.location_ids),
            "persons": sorted(policy.person_ids),
            "user_destination_value": policy.user_value,
        },
        "output_schema_example": schema,
        "task": {
            "original_text": original_text,
            "intent": intent,
            "entities": dict(entities),
        },
    }
    if correction_errors:
        payload["previous_validation_errors"] = list(correction_errors)
        payload["correction_instruction"] = (
            "Your previous JSON failed validation. Return a corrected JSON "
            "plan that fixes every listed error."
        )
    return json.dumps(payload, indent=2, sort_keys=True)
