"""Shared dialogue fixtures."""

from __future__ import annotations

import pytest

from src.dialogue import DialogueManager
from src.nlu.dataset import load_train
from src.nlu.nlu import NLUEngine


@pytest.fixture(scope="session")
def nlu_engine():
    return NLUEngine.from_examples(load_train())


@pytest.fixture
def manager(nlu_engine):
    return DialogueManager(engine=nlu_engine)
