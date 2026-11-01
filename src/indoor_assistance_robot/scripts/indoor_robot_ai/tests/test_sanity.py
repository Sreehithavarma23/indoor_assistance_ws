"""Sanity check that the Phase 1 environment imports and loads config."""

from src.config_loader import load_intents, load_locations, load_objects
from src.logging_utils import setup_logging


def test_project_environment():
    logger = setup_logging()
    logger.info("Phase 1 sanity check")

    intents_config = load_intents()
    objects_config = load_objects()
    locations_config = load_locations()

    assert intents_config["intents"] == [
        "NAVIGATE",
        "RETRIEVE",
        "DELIVER",
        "FOLLOW",
        "STOP",
        "RETURN",
    ]
    objects = objects_config["objects"]
    locations = locations_config["locations"]
    assert objects and all("id" in item for item in objects)
    assert locations and all("id" in item for item in locations)
    assert logger.name == "indoor_robot_ai"
