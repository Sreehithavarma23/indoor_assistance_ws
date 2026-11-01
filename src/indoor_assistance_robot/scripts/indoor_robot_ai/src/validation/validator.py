"""Validate an NLUResult. Does not parse text, plan, or execute."""

from __future__ import annotations

from typing import Optional

from src.logging_utils import get_logger
from src.nlu.types import NLUResult
from src.validation.exceptions import ValidationError
from src.validation.rules import (
    competing_robot_intents,
    conflicting_mention_values,
    missing_required_slots,
    unknown_catalog_reference,
    unknown_catalog_values,
    unresolved_references,
    unsupported_capability_request,
)
from src.validation.schema import IntentRequirements, ValidationPolicy, load_policy
from src.validation.types import ValidationResult, ValidationStatus

logger = get_logger("validation")


def _empty_spec() -> IntentRequirements:
    return IntentRequirements()


def validate(
    nlu_result: NLUResult,
    policy: Optional[ValidationPolicy] = None,
) -> ValidationResult:
    """Decide VALID / INCOMPLETE / AMBIGUOUS / UNSUPPORTED.

    Priority (first match wins):
    1. Intent is not an executable robot skill → UNSUPPORTED
    2. Configured out-of-skill capability cue → UNSUPPORTED
    3. A filled required slot is not in the catalog → UNSUPPORTED
    4. An explicit but ungrounded object/destination → UNSUPPORTED
    5. Competing robot-intent probabilities → AMBIGUOUS
    6. Conflicting grounded values or unresolved deixis → AMBIGUOUS
    7. Required slots missing → INCOMPLETE
    8. Otherwise → VALID
    """
    if nlu_result is None or not isinstance(nlu_result, NLUResult):
        raise ValidationError("validate() requires an NLUResult.")

    policy = policy or load_policy()
    spec = policy.requirements.get(nlu_result.intent, _empty_spec())

    if nlu_result.intent not in policy.robot_intents:
        result = ValidationResult.from_nlu(
            nlu_result,
            ValidationStatus.UNSUPPORTED,
            reasons=[
                f"Intent '{nlu_result.intent}' is not an executable robot skill."
            ],
        )
        logger.info("Validation status=%s intent=%s", result.status.value, result.intent)
        return result

    unsupported_cue, cue_reason = unsupported_capability_request(nlu_result)
    if unsupported_cue:
        result = ValidationResult.from_nlu(
            nlu_result,
            ValidationStatus.UNSUPPORTED,
            reasons=[cue_reason],
        )
        logger.info("Validation status=%s intent=%s", result.status.value, result.intent)
        return result

    unknown = unknown_catalog_values(nlu_result.entities, spec, policy)
    if unknown:
        result = ValidationResult.from_nlu(
            nlu_result,
            ValidationStatus.UNSUPPORTED,
            reasons=unknown,
        )
        logger.info("Validation status=%s intent=%s", result.status.value, result.intent)
        return result

    unknown_reference = unknown_catalog_reference(nlu_result)
    if unknown_reference:
        result = ValidationResult.from_nlu(
            nlu_result,
            ValidationStatus.UNSUPPORTED,
            reasons=[unknown_reference],
        )
        logger.info("Validation status=%s intent=%s", result.status.value, result.intent)
        return result

    competing, compete_reason = competing_robot_intents(nlu_result, policy)
    if competing:
        result = ValidationResult.from_nlu(
            nlu_result,
            ValidationStatus.AMBIGUOUS,
            reasons=[compete_reason],
        )
        logger.info("Validation status=%s intent=%s", result.status.value, result.intent)
        return result

    conflict_slots, conflict_reasons = conflicting_mention_values(
        nlu_result.mentions, spec
    )
    deixis_slots, deixis_reasons = unresolved_references(
        nlu_result.original_text,
        nlu_result.entities,
        spec,
        policy,
    )
    ambiguous = list(dict.fromkeys(conflict_slots + deixis_slots))
    amb_reasons = conflict_reasons + deixis_reasons
    if ambiguous:
        result = ValidationResult.from_nlu(
            nlu_result,
            ValidationStatus.AMBIGUOUS,
            reasons=amb_reasons,
            ambiguous_entities=ambiguous,
        )
        logger.info("Validation status=%s intent=%s", result.status.value, result.intent)
        return result

    missing, missing_reasons = missing_required_slots(nlu_result.entities, spec)
    if missing:
        result = ValidationResult.from_nlu(
            nlu_result,
            ValidationStatus.INCOMPLETE,
            reasons=missing_reasons,
            missing_entities=missing,
        )
        logger.info("Validation status=%s intent=%s", result.status.value, result.intent)
        return result

    result = ValidationResult.from_nlu(nlu_result, ValidationStatus.VALID)
    logger.info("Validation status=%s intent=%s", result.status.value, result.intent)
    return result
