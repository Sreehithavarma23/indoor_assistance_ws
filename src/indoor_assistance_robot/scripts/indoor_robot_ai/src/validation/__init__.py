"""Phase 4 task validation: NLUResult → VALID / INCOMPLETE / AMBIGUOUS / UNSUPPORTED.

    from src.nlu import parse
    from src.validation import validate

    result = validate(parse("Bring me the bottle."))
"""

from src.validation.types import ValidationResult, ValidationStatus
from src.validation.validator import validate

__all__ = ["ValidationResult", "ValidationStatus", "validate"]
