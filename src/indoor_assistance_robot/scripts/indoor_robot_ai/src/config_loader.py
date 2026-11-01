"""Load YAML files from the project's config/ directory."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def load_yaml(filename: str) -> dict:
    """Read a YAML file from config/ and return a dict.

    Missing or empty files yield an empty dict so callers can fill catalogs later.
    """
    path = CONFIG_DIR / filename
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def load_intents() -> dict:
    """Load config/intents.yaml."""
    return load_yaml("intents.yaml")


def load_objects() -> dict:
    """Load config/objects.yaml."""
    return load_yaml("objects.yaml")


def load_locations() -> dict:
    """Load config/locations.yaml."""
    return load_yaml("locations.yaml")


def load_speech() -> dict:
    """Load config/speech.yaml."""
    return load_yaml("speech.yaml")


def load_persons() -> dict:
    """Load config/persons.yaml."""
    return load_yaml("persons.yaml")


def load_nlu() -> dict:
    """Load config/nlu.yaml."""
    return load_yaml("nlu.yaml")


def load_validation() -> dict:
    """Load config/validation.yaml."""
    return load_yaml("validation.yaml")


def load_dialogue() -> dict:
    """Load config/dialogue.yaml."""
    return load_yaml("dialogue.yaml")


def load_planning() -> dict:
    """Load config/planning.yaml."""
    return load_yaml("planning.yaml")


def load_llm_planning() -> dict:
    """Load config/llm_planning.yaml."""
    return load_yaml("llm_planning.yaml")


def load_pipeline() -> dict:
    """Load config/pipeline.yaml."""
    return load_yaml("pipeline.yaml")
