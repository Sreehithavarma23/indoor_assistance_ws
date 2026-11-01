"""Intent classifier B: pretrained sentence embeddings + logistic regression.

Uses Sentence-Transformers when available. Embeddings are pretrained;
only the logistic regression head is fit on project training data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.config_loader import PROJECT_ROOT, load_nlu
from src.logging_utils import get_logger
from src.nlu.exceptions import IntentModelError
from src.nlu.types import IntentPrediction

logger = get_logger("nlu.embedding")

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "models" / "nlu" / "embedding_minilm"


@dataclass
class EmbeddingLogRegIntentClassifier:
    """Frozen encoder + trained logistic regression over intent labels."""

    encoder_name: str
    labels: List[str]
    unsupported_threshold: float
    classifier: LogisticRegression
    _encoder: object = None

    def _ensure_encoder(self):
        if self._encoder is not None:
            return self._encoder
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise IntentModelError(
                "sentence-transformers is required for the embedding classifier."
            ) from exc
        logger.info("Loading sentence encoder %s", self.encoder_name)
        self._encoder = SentenceTransformer(self.encoder_name)
        return self._encoder

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        encoder = self._ensure_encoder()
        vectors = encoder.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def predict(self, text: str) -> IntentPrediction:
        raw = (text or "").strip()
        if not raw:
            zeros = {label: 0.0 for label in self.labels}
            return IntentPrediction(
                intent="UNSUPPORTED",
                confidence=0.0,
                probabilities=zeros,
            )
        vector = self.embed([raw])
        probabilities = self.classifier.predict_proba(vector)[0]
        mapping = {
            str(label): float(prob)
            for label, prob in zip(self.classifier.classes_, probabilities)
        }
        for label in self.labels:
            mapping.setdefault(label, 0.0)
        intent = max(mapping, key=mapping.get)
        confidence = mapping[intent]
        if confidence < self.unsupported_threshold:
            logger.info(
                "Embedding intent %s confidence %.3f below %.2f.",
                intent,
                confidence,
                self.unsupported_threshold,
            )
        return IntentPrediction(
            intent=intent,
            confidence=confidence,
            probabilities=mapping,
        )

    def save(self, directory: Optional[Path] = None) -> Path:
        out = Path(directory or DEFAULT_ARTIFACT_DIR)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.classifier, out / "logreg.joblib")
        meta = {
            "encoder_name": self.encoder_name,
            "labels": self.labels,
            "unsupported_threshold": self.unsupported_threshold,
            "pretrained_encoder": True,
            "fine_tuned_encoder": False,
            "trained_head": "LogisticRegression",
        }
        (out / "meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        logger.info("Wrote embedding classifier to %s", out)
        return out

    @classmethod
    def load(cls, directory: Optional[Path] = None) -> "EmbeddingLogRegIntentClassifier":
        out = Path(directory or DEFAULT_ARTIFACT_DIR)
        meta_path = out / "meta.json"
        model_path = out / "logreg.joblib"
        if not meta_path.is_file() or not model_path.is_file():
            raise IntentModelError(f"Embedding classifier not found under {out}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(
            encoder_name=str(meta["encoder_name"]),
            labels=list(meta["labels"]),
            unsupported_threshold=float(meta["unsupported_threshold"]),
            classifier=joblib.load(model_path),
        )


def train_embedding_classifier(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    encoder_name: str = DEFAULT_MODEL_NAME,
    random_state: int = 42,
) -> EmbeddingLogRegIntentClassifier:
    if len(texts) != len(labels) or not texts:
        raise IntentModelError("Training requires a non-empty aligned text/label set.")
    config = load_nlu()
    class_labels = list(config.get("classifier_labels", []))
    threshold = float(config.get("unsupported_confidence_threshold", 0.40))

    model = EmbeddingLogRegIntentClassifier(
        encoder_name=encoder_name,
        labels=class_labels,
        unsupported_threshold=threshold,
        classifier=LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
    )
    X = model.embed(texts)
    model.classifier.fit(X, list(labels))
    return model


def embedding_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False
