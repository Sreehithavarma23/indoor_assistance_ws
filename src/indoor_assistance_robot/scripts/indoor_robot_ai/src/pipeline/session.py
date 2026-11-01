"""Conversation session wrapping a single DialogueManager.

Each PipelineSession owns its dialogue state. Create a new session for each
independent conversation so clarification state cannot leak across users.
"""

from __future__ import annotations

from typing import Optional

from src.dialogue import DialogueManager
from src.nlu.nlu import NLUEngine


class PipelineSession:
    """Stateful conversation handle for the end-to-end pipeline.

    Usage::

        session = PipelineSession()
        r1 = pipeline.run_text("Bring me the bottle.", session=session)
        r2 = pipeline.run_text("From the kitchen.", session=session)
        session.reset()  # clear pending clarification / history
    """

    def __init__(self, engine: Optional[NLUEngine] = None) -> None:
        self.engine = engine
        self.dialogue = DialogueManager(engine=engine)

    def reset(self) -> None:
        """Clear pending clarification and turn history for this session."""
        self.dialogue.reset()
