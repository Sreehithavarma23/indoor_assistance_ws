"""Shared NLU fixtures. Trains only on the train split."""

from __future__ import annotations

import pytest

from src.nlu.dataset import assert_no_overlap, load_test, load_train
from src.nlu.nlu import NLUEngine


@pytest.fixture(scope="session")
def train_examples():
    return load_train()


@pytest.fixture(scope="session")
def test_examples():
    return load_test()


@pytest.fixture(scope="session")
def nlu_engine(train_examples, test_examples):
    assert_no_overlap(train_examples, test_examples)
    return NLUEngine.from_examples(train_examples)
