"""Phase 3 NLU: text → intent + entities.

Public interface:

    from src.nlu import parse
    result = parse("Can you take me to the kitchen?")
    result.intent
    result.entities

This package does not validate tasks, plan actions, or call ROS 2.
"""

from src.nlu.nlu import NLUEngine, parse, parse_transcript
from src.nlu.types import EntityMention, NLUResult

__all__ = [
    "EntityMention",
    "NLUEngine",
    "NLUResult",
    "parse",
    "parse_transcript",
]
