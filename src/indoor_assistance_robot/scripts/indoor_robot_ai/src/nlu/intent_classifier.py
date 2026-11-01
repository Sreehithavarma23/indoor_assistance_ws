"""Replaceable intent-classification baseline: TF-IDF + Logistic Regression.

This is a supervised model trained on the text dataset. It is not a
sentence lookup table and not an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol, Sequence

import joblib
import hashlib
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config_loader import load_nlu
from src.logging_utils import get_logger
from src.nlu.exceptions import IntentModelError
from src.nlu.types import IntentPrediction

logger = get_logger("nlu.intent")


class IntentClassifier(Protocol):
    """Public contract so a later transformer/LLM backend can replace this."""

    def predict(self, text: str) -> IntentPrediction:
        ...


@dataclass
class TfidfLogRegIntentClassifier:
    """TF-IDF unigrams+bigrams with class-balanced logistic regression."""

    pipeline: Pipeline
    labels: List[str]
    unsupported_threshold: float
    training_metadata: Dict[str, Any] | None = None

    def predict(self, text: str) -> IntentPrediction:
        raw = prepare_intent_text(text)
        if not raw:
            zeros = {label: 0.0 for label in self.labels}
            return IntentPrediction(
                intent="UNSUPPORTED",
                confidence=0.0,
                probabilities=zeros,
            )

        probabilities = self.pipeline.predict_proba([raw])[0]
        mapping = {
            str(label): float(prob)
            for label, prob in zip(self.pipeline.classes_, probabilities)
        }
        for label in self.labels:
            mapping.setdefault(label, 0.0)
        intent = max(mapping, key=mapping.get)
        confidence = mapping[intent]
        # Confidence is reported and logged only. Rewriting the argmax label
        # to UNSUPPORTED when confidence < threshold hid correct decisions on
        # paraphrases (7-way softmax is diffuse). Narrow STOP→NAVIGATE recovery
        # and capability-cue checks live in NLUEngine.parse instead.
        if confidence < self.unsupported_threshold:
            logger.info(
                "Intent %s confidence %.3f is below %.2f (signal for later validation).",
                intent,
                confidence,
                self.unsupported_threshold,
            )
        return IntentPrediction(
            intent=intent,
            confidence=confidence,
            probabilities=mapping,
        )

    def save(self, path: Path, metadata: Dict[str, Any] | None = None) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "labels": self.labels,
                "unsupported_threshold": self.unsupported_threshold,
                "metadata": metadata or self.training_metadata or {},
            },
            path,
        )
        logger.info("Wrote intent model to %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "TfidfLogRegIntentClassifier":
        if not path.is_file():
            raise IntentModelError(f"Intent model not found: {path}")
        provenance_path = path.with_suffix(path.suffix + ".provenance.json")
        provenance: Dict[str, Any] = {}
        if provenance_path.is_file():
            try:
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except (OSError, json.JSONDecodeError) as exc:
                raise IntentModelError(
                    f"Could not read artifact provenance: {provenance_path}"
                ) from exc
            expected = provenance.get("artifact_sha256")
            if expected and digest != expected:
                raise IntentModelError(
                    f"Artifact SHA-256 mismatch for {path}: expected {expected}, got {digest}"
                )
        payload = joblib.load(path)
        embedded = dict(payload.get("metadata") or {})
        if provenance and embedded != {
            key: value for key, value in provenance.items()
            if key not in {"artifact_path", "artifact_sha256"}
        }:
            # The sidecar carries the same training record plus the artifact
            # digest/path; compare the embedded provenance's shared fields.
            for key, value in embedded.items():
                if provenance.get(key) != value:
                    raise IntentModelError(
                        f"Artifact metadata mismatch for {path}: field {key!r}"
                    )
        return cls(
            pipeline=payload["pipeline"],
            labels=list(payload["labels"]),
            unsupported_threshold=float(payload["unsupported_threshold"]),
            training_metadata=embedded,
        )


def prepare_intent_text(text: str) -> str:
    """Shared text preparation used by both fitting and prediction."""
    return str(text or "").strip()


def train_intent_classifier(
    texts: Sequence[str],
    labels: Sequence[str],
    random_state: int = 42,
) -> TfidfLogRegIntentClassifier:
    """Fit the baseline on the provided split only."""
    if len(texts) != len(labels) or not texts:
        raise IntentModelError("Training requires a non-empty aligned text/label set.")

    config = load_nlu()
    class_labels = list(config.get("classifier_labels", []))
    threshold = float(config.get("unsupported_confidence_threshold", 0.40))

    # No English stop-word list: words such as "to", "me", and "stop" carry intent.
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    lowercase=True,
                    min_df=1,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit([prepare_intent_text(text) for text in texts], list(labels))
    logger.info("Trained TF-IDF+LR intent classifier on %d examples.", len(texts))
    return TfidfLogRegIntentClassifier(
        pipeline=pipeline,
        labels=class_labels,
        unsupported_threshold=threshold,
        training_metadata={
            "random_state": random_state,
            "preprocessing": {
                "strip_outer_whitespace": True,
                "vectorizer_lowercase": True,
                "vectorizer_strip_accents": "unicode",
                "vectorizer_ngram_range": [1, 2],
                "vectorizer_min_df": 1,
            },
            "logistic_regression": {
                "max_iter": 1000,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "random_state": random_state,
            },
        },
    )


def default_model_path() -> Path:
    from src.config_loader import PROJECT_ROOT

    return PROJECT_ROOT / "models" / "nlu" / "intent_tfidf_lr.joblib"
