"""Phase 5 dialogue: clarification + context on top of NLU and validation.

    from src.dialogue import DialogueManager

    manager = DialogueManager()
    result = manager.process("Bring me the bottle.")
    # result.status == CLARIFICATION_REQUIRED
    result = manager.process("From the kitchen.")
    # result.status == READY
"""

from src.dialogue.dialogue_manager import DialogueManager
from src.dialogue.types import DialogueResult, DialogueStatus

__all__ = [
    "DialogueManager",
    "DialogueResult",
    "DialogueStatus",
]
