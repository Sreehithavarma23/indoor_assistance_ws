"""Fixtures for typed language-understanding tests."""

from __future__ import annotations

import pytest

from src.nlu.dataset import load_train
from src.nlu.nlu import NLUEngine
from src.pipeline import AIPipeline


@pytest.fixture(scope="session")
def nlu_engine():
    return NLUEngine.from_examples(load_train())


@pytest.fixture
def pipeline(nlu_engine):
    return AIPipeline(engine=nlu_engine)
