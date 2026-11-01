"""Errors for constrained LLM-assisted planning."""


class LLMPlanningError(Exception):
    """Base error for Phase 7 LLM planning."""


class LLMPlanningInputError(LLMPlanningError):
    """Dialogue/task input is not eligible for LLM planning."""


class ProviderError(LLMPlanningError):
    """LLM provider failed (network, HTTP, or unexpected response)."""


class MissingCredentialsError(ProviderError):
    """Required API credentials are not configured."""


class ProviderTimeoutError(ProviderError):
    """Provider request timed out."""


class ResponseParseError(LLMPlanningError):
    """Model output could not be parsed into a structured plan."""


class GroundingError(LLMPlanningError):
    """Plan references unknown catalog entities or invalid parameters."""
