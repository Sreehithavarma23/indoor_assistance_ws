"""Minimal console logging for the project."""

import logging
import sys

LOGGER_NAME = "indoor_robot_ai"


def get_logger(name: str = "") -> logging.Logger:
    """Return the project logger or a child such as 'speech.asr'."""
    if not name:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the project logger.

    Safe to call more than once: extra handlers are not added.
    File logging is not configured yet.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
