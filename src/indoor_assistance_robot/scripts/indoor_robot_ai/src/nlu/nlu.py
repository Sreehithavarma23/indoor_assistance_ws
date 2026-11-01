"""Unified NLU entry point: text → intent + entities.

Accepts plain strings so typed text and ASR transcripts use the same path.
Does not validate tasks, ask questions, or plan robot actions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

from src.logging_utils import get_logger, setup_logging
from src.nlu.cues import (
    has_contradictory_instruction,
    has_negated_movement,
    matches_unsupported_capability,
)
from src.nlu.dataset import load_train, texts_and_intents
from src.nlu.entity_extractor import extract_entities
from src.nlu.exceptions import NLUError
from src.nlu.intent_classifier import (
    IntentClassifier,
    TfidfLogRegIntentClassifier,
    default_model_path,
    train_intent_classifier,
)
from src.nlu.types import NLUExample, NLUResult

logger = get_logger("nlu")

def _action_intent_correction(
    prediction,
    entities: dict,
    unsupported_threshold: float,
    text: str,
) -> Optional[str]:
    """Recover a clear, bare positive navigation command.

    A bare deictic command such as "Go there." can be NAVIGATE even when
    the classifier predicts STOP or UNSUPPORTED. Destination resolution
    remains the validator/dialogue manager's responsibility.
    """
    if has_negated_movement(text):
        return None

    if has_contradictory_instruction(text):
        return None

    if matches_unsupported_capability(text):
        return None

    action = entities.get("action")
    if action != "navigate":
        return None

    # Only recover bare action commands. Preserve classifier decisions for
    # utterances containing additional slots.
    if any(slot != "action" for slot in entities):
        return None

    # Recover only the two known failure modes, not arbitrary intents.
    if prediction.intent == "STOP":
        if prediction.confidence >= unsupported_threshold:
            return None
    elif prediction.intent != "UNSUPPORTED":
        return None

    candidate = str(action).upper()
    if candidate in prediction.probabilities:
        return candidate

    return None

def _strip_positive_navigate(entities: dict) -> dict:
    """Remove navigate action after a negation so STOP stays entity-light."""
    if entities.get("action") != "navigate":
        return entities
    cleaned = dict(entities)
    cleaned.pop("action", None)
    return cleaned

class NLUEngine:
    """Intent classifier + entity extractor behind one parse() method."""

    def __init__(self, classifier: IntentClassifier) -> None:
        self.classifier = classifier

    def parse(self, text: str) -> NLUResult:
        if text is None or not isinstance(text, str):
            raise NLUError("NLU.parse requires a text string.")

        original = text.strip()
        prediction = self.classifier.predict(original)
        entities, mentions = extract_entities(original, intent=prediction.intent)
        intent = prediction.intent
        confidence = prediction.confidence
        threshold = float(getattr(self.classifier, "unsupported_threshold", 0.40))
        probabilities = dict(prediction.probabilities)

        if has_contradictory_instruction(original):
            intent = "UNSUPPORTED"
            confidence = max(probabilities.get("UNSUPPORTED", 0.0), confidence)
            entities, mentions = extract_entities(original, intent=intent)
            logger.info("Contradictory motion instruction → UNSUPPORTED.")
        elif has_negated_movement(original):
            # Negated movement is a stop command, never positive navigation.
            intent = "STOP"
            confidence = max(
                probabilities.get("STOP", 0.0),
                prediction.confidence if prediction.intent == "STOP" else 0.0,
            )
            if confidence <= 0.0:
                confidence = probabilities.get("STOP", 0.5) or 0.5
            entities, mentions = extract_entities(original, intent=intent)
            entities = _strip_positive_navigate(entities)
            logger.info("Interpreted negated movement as STOP.")
        elif matches_unsupported_capability(original):
            # Clear out-of-skill cues override a mistaken robot intent
            # (including low-confidence STOP on e.g. "Do a backflip.").
            # Does not use low confidence alone as the decision rule.
            intent = "UNSUPPORTED"
            confidence = max(probabilities.get("UNSUPPORTED", 0.0), confidence)
            entities, mentions = extract_entities(original, intent=intent)
            logger.info("Matched unsupported capability cue; intent=UNSUPPORTED.")
        else:
            corrected_intent = _action_intent_correction(
                prediction, entities, threshold, original
            )
            if corrected_intent is not None:
                intent = corrected_intent
                confidence = probabilities.get(
                    corrected_intent, prediction.confidence
                )
                entities, mentions = extract_entities(original, intent=intent)
                logger.info(
                    "Corrected low-confidence STOP to %s from action alias.",
                    corrected_intent,
                )

        result = NLUResult(
            original_text=original,
            intent=intent,
            intent_confidence=confidence,
            entities=entities,
            mentions=mentions,
            intent_probabilities=probabilities,
        )
        logger.info(
            "NLU intent=%s confidence=%.3f entities=%s",
            result.intent,
            result.intent_confidence,
            result.entities,
        )
        return result

    def save(self, path: Optional[Path] = None) -> Path:
        if not isinstance(self.classifier, TfidfLogRegIntentClassifier):
            raise NLUError("Only the TF-IDF baseline can be serialized here.")
        return self.classifier.save(path or default_model_path())

    @classmethod
    def from_examples(cls, examples: Sequence[NLUExample]) -> "NLUEngine":
        texts, labels = texts_and_intents(examples)
        classifier = train_intent_classifier(texts, labels)
        return cls(classifier)

    @classmethod
    def train_default(cls) -> "NLUEngine":
        return cls.from_examples(load_train())

    @classmethod
    def load(cls, path: Optional[Path] = None, backend: Optional[str] = None) -> "NLUEngine":
        """Load a classifier backend.

        - Default / explicit tfidf path: Phase-3 TF-IDF artifact.
        - backend='embedding' or config intent_backend=embedding: MiniLM head.
        """
        from src.nlu.backends import resolve_intent_classifier

        if path is not None:
            return cls(TfidfLogRegIntentClassifier.load(path))
        try:
            return cls(resolve_intent_classifier(backend))
        except Exception as exc:
            if isinstance(exc, NLUError):
                raise
            raise NLUError(str(exc)) from exc

    @classmethod
    def load_or_train(cls, path: Optional[Path] = None) -> "NLUEngine":
        """Deprecated compatibility alias; inference never trains implicitly."""
        return cls.load(path=path)


def parse(text: str, engine: Optional[NLUEngine] = None) -> NLUResult:
    """Public functional interface for later ASR → NLU wiring."""
    setup_logging()
    nlu = engine or NLUEngine.load()
    return nlu.parse(text)


def parse_transcript(transcript: Union[str, object], engine: Optional[NLUEngine] = None) -> NLUResult:
    """Accept Transcript.text or a raw string. Does not call ASR."""
    if hasattr(transcript, "text"):
        return parse(str(getattr(transcript, "text")), engine=engine)
    return parse(str(transcript), engine=engine)
