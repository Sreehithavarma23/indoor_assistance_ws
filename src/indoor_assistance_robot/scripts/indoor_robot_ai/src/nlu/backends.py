"""Stable factory for replaceable intent backends."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.config_loader import PROJECT_ROOT, load_nlu
from src.nlu.embedding_classifier import (
    DEFAULT_ARTIFACT_DIR,
    EmbeddingLogRegIntentClassifier,
    embedding_available,
)
from src.nlu.exceptions import IntentModelError
from src.nlu.intent_classifier import (
    IntentClassifier,
    TfidfLogRegIntentClassifier,
    default_model_path,
)


def resolve_intent_classifier(
    backend: Optional[str] = None,
) -> IntentClassifier:
    """Load the configured intent backend without changing NLUEngine.parse."""
    config = load_nlu()
    name = (backend or config.get("intent_backend") or "tfidf").strip().lower()
    if name in {"tfidf", "tfidf_logreg", "baseline"}:
        path = Path(config.get("tfidf_model_path") or default_model_path())
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise IntentModelError(f"TF-IDF model missing: {path}")
        return TfidfLogRegIntentClassifier.load(path)
    if name in {"embedding", "minilm", "minilm_logreg"}:
        if not embedding_available():
            raise IntentModelError("sentence-transformers is not installed.")
        directory = Path(config.get("embedding_model_dir") or DEFAULT_ARTIFACT_DIR)
        return EmbeddingLogRegIntentClassifier.load(directory)
    raise IntentModelError(f"Unknown intent backend: {name}")
