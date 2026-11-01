"""Errors raised by the planning module."""


class PlanningError(Exception):
    """Base error for high-level task planning."""


class PlanningInputError(PlanningError):
    """Dialogue/task input is not eligible for planning."""


class PlanValidationError(PlanningError):
    """Generated plan failed structural validation."""
