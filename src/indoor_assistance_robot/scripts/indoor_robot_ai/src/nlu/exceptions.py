"""Errors raised by the NLU module only."""


class NLUError(Exception):
    """Base error for natural-language understanding."""


class DatasetError(NLUError):
    """Train/test data is missing or malformed."""


class IntentModelError(NLUError):
    """The intent classifier is missing or cannot be used."""
