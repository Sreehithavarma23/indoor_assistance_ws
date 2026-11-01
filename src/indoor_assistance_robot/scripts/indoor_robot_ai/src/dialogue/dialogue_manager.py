"""DialogueManager: NLU → validate → clarify / merge → re-validate.

Does not plan robot actions. Does not call an LLM.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from src.config_loader import load_dialogue
from src.dialogue.clarifier import generate_clarification, pending_slots_from_validation
from src.dialogue.exceptions import DialogueError
from src.dialogue.resolver import (
    ResolveAction,
    build_merged_nlu,
    decide_follow_up,
    merge_entities,
)
from src.dialogue.types import (
    DialogueResult,
    DialogueState,
    DialogueStatus,
    DialogueTurn,
    task_from_nlu,
)
from src.logging_utils import get_logger, setup_logging
from src.nlu.nlu import NLUEngine, parse
from src.nlu.types import NLUResult
from src.validation.types import ValidationResult, ValidationStatus
from src.validation.validator import validate

logger = get_logger("dialogue")


@lru_cache(maxsize=1)
def _dialogue_config() -> dict:
    return load_dialogue()


class DialogueManager:
    """Stateful multi-turn dialogue on top of Phase 3 and Phase 4."""

    def __init__(self, engine: Optional[NLUEngine] = None) -> None:
        setup_logging()
        self.engine = engine
        self.state = DialogueState()

    def reset(self) -> None:
        self.state.reset()

    def _parse(self, text: str) -> NLUResult:
        return parse(text, engine=self.engine)

    def _history_append(self, role: str, text: str) -> None:
        size = int(_dialogue_config().get("history_size", 8))
        self.state.turns.append(DialogueTurn(role=role, text=text))
        if len(self.state.turns) > size:
            self.state.turns = self.state.turns[-size:]

    def _max_turns(self) -> int:
        return int(_dialogue_config().get("max_clarification_turns", 3))

    def _result(
        self,
        status: DialogueStatus,
        nlu: Optional[NLUResult],
        validation: Optional[ValidationResult],
        question: Optional[str] = None,
        pending: Optional[List[str]] = None,
        message: str = "",
        original_text: str = "",
    ) -> DialogueResult:
        return DialogueResult(
            status=status,
            question=question,
            pending_slots=list(pending or []),
            task=task_from_nlu(nlu),
            validation=validation,
            clarification_count=self.state.clarification_count,
            original_text=original_text or (nlu.original_text if nlu else ""),
            message=message,
        )

    def _ask_clarification(
        self,
        nlu: NLUResult,
        validation: ValidationResult,
    ) -> DialogueResult:
        pending = pending_slots_from_validation(validation)
        if self.state.clarification_count >= self._max_turns():
            self.state.awaiting_clarification = False
            self.state.pending_slots = pending
            self.state.last_validation = validation
            self.state.active_nlu = nlu
            logger.info("Clarification failed after %d turns", self.state.clarification_count)
            return self._result(
                DialogueStatus.CLARIFICATION_FAILED,
                nlu,
                validation,
                pending=pending,
                message="Maximum clarification turns exceeded.",
            )

        question = generate_clarification(nlu, validation, pending)
        self.state.clarification_count += 1
        self.state.awaiting_clarification = True
        self.state.active_nlu = nlu
        self.state.last_validation = validation
        self.state.pending_slots = pending
        self.state.last_question = question
        self._history_append("system", question)
        logger.info(
            "Clarification required count=%d pending=%s q=%r",
            self.state.clarification_count,
            pending,
            question,
        )
        return self._result(
            DialogueStatus.CLARIFICATION_REQUIRED,
            nlu,
            validation,
            question=question,
            pending=pending,
            message=question,
        )

    def _handle_validated(
        self,
        nlu: NLUResult,
        validation: ValidationResult,
    ) -> DialogueResult:
        if validation.status == ValidationStatus.UNSUPPORTED:
            self.state.reset()
            self.state.active_nlu = nlu
            self.state.last_validation = validation
            return self._result(
                DialogueStatus.UNSUPPORTED,
                nlu,
                validation,
                message="Request is outside supported robot capabilities.",
            )

        if validation.status == ValidationStatus.VALID:
            self.state.awaiting_clarification = False
            self.state.active_nlu = nlu
            self.state.last_validation = validation
            self.state.pending_slots = []
            self.state.last_question = None
            # Keep clarification_count for inspection but clear await flag.
            return self._result(
                DialogueStatus.READY,
                nlu,
                validation,
                message="Task is ready for planning.",
            )

        # INCOMPLETE or AMBIGUOUS
        return self._ask_clarification(nlu, validation)

    def process(self, text: str, nlu: Optional[NLUResult] = None) -> DialogueResult:
        """Process one user utterance. Optional nlu injects a constructed Phase 3 result."""
        if text is None or not isinstance(text, str):
            raise DialogueError("DialogueManager.process requires a text string.")
        utterance = text.strip()
        if not utterance and nlu is None:
            raise DialogueError("Empty utterance.")

        self._history_append("user", utterance or (nlu.original_text if nlu else ""))

        if self.state.awaiting_clarification and self.state.active_nlu is not None:
            return self._process_follow_up(utterance, injected=nlu)

        parsed = nlu if nlu is not None else self._parse(utterance)
        validation = validate(parsed)
        # Fresh task: reset clarification counter for a new dialogue thread.
        self.state.clarification_count = 0
        self.state.active_nlu = parsed
        self.state.last_validation = validation
        return self._handle_validated(parsed, validation)

    def _process_follow_up(
        self,
        utterance: str,
        injected: Optional[NLUResult] = None,
    ) -> DialogueResult:
        active = self.state.active_nlu
        assert active is not None
        pending = list(self.state.pending_slots)
        new_nlu = injected if injected is not None else self._parse(utterance)
        new_validation = validate(new_nlu)
        decision = decide_follow_up(utterance, active, pending, new_nlu, new_validation)
        logger.info("Follow-up decision=%s reason=%s", decision.action.value, decision.reason)

        if decision.action == ResolveAction.INTERRUPT:
            cancelled = new_nlu
            self.state.reset()
            self.state.active_nlu = cancelled
            self.state.last_validation = new_validation
            return self._result(
                DialogueStatus.CANCELLED,
                cancelled,
                new_validation,
                message="Pending task cancelled by STOP.",
            )

        if decision.action == ResolveAction.REPLACE:
            self.state.clarification_count = 0
            self.state.active_nlu = new_nlu
            self.state.last_validation = new_validation
            self.state.awaiting_clarification = False
            self.state.pending_slots = []
            self.state.last_question = None
            return self._handle_validated(new_nlu, new_validation)

        if decision.action == ResolveAction.MERGE:
            merged_entities = merge_entities(active.entities, decision.slot_updates)
            merged_nlu = build_merged_nlu(active, merged_entities, utterance)
            merged_validation = validate(merged_nlu)
            self.state.active_nlu = merged_nlu
            self.state.last_validation = merged_validation
            return self._handle_validated(merged_nlu, merged_validation)

        # UNRESOLVED: count as a failed clarification attempt via re-ask path.
        assert self.state.last_validation is not None
        return self._ask_clarification(active, self.state.last_validation)
