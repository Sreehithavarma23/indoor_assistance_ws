"""Validation tests reuse a train-split NLU engine."""

from __future__ import annotations

import pytest

from src.nlu.dataset import load_train
from src.nlu.nlu import NLUEngine
from src.nlu.types import EntityMention, NLUResult


@pytest.fixture(scope="session")
def nlu_engine():
    return NLUEngine.from_examples(load_train())


def make_nlu(
    text: str,
    intent: str,
    entities: dict | None = None,
    confidence: float = 0.8,
    probabilities: dict | None = None,
    mentions: list | None = None,
) -> NLUResult:
    probs = probabilities or {intent: confidence}
    return NLUResult(
        original_text=text,
        intent=intent,
        intent_confidence=confidence,
        entities=entities or {},
        mentions=mentions or [],
        intent_probabilities=probs,
    )


def mention(role: str, value: str, entity_type: str | None = None) -> EntityMention:
    return EntityMention(
        entity_type=entity_type or role,
        text=value,
        value=value,
        role=role,
    )
