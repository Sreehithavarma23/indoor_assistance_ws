"""Smoke tests for NLU v2 dataset artifacts (run after build)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config_loader import PROJECT_ROOT
from src.data_pipeline.schema import SPLIT_NAMES, VALID_SOURCES
from src.data_pipeline.split import check_leakage, load_split, split_records
from src.data_pipeline.generate import generate_all
from src.nlu.normalization import normalize_text

SPLITS_DIR = PROJECT_ROOT / "data" / "nlu_v2" / "splits"


@pytest.fixture(scope="module")
def built_splits():
    if not (SPLITS_DIR / "train.jsonl").is_file():
        pytest.skip("NLU v2 splits not built yet; run: PYTHONPATH= python -m src.data_pipeline")
    return {name: load_split(name) for name in SPLIT_NAMES}


def test_generate_produces_diverse_sources():
    records = generate_all(seed=0, max_per_family=2)
    assert len(records) >= 400
    sources = {r.source for r in records}
    assert "synthetic_template" in sources
    assert "legacy_hand_authored" in sources or "human_authored" in sources
    for r in records:
        assert r.intent in {
            "NAVIGATE",
            "RETRIEVE",
            "DELIVER",
            "FOLLOW",
            "STOP",
            "RETURN",
            "UNSUPPORTED",
        }
        assert r.source in VALID_SOURCES
        assert r.family_id


def test_family_split_has_no_train_test_family_overlap():
    records = generate_all(seed=1, max_per_family=2)
    splits = split_records(records, seed=1)
    issues = check_leakage(splits)
    family_issues = [i for i in issues if i.startswith("Family overlap")]
    assert not family_issues, family_issues


def test_built_splits_cover_intents(built_splits):
    train_intents = {r.intent for r in built_splits["train"]}
    for intent in ("NAVIGATE", "RETRIEVE", "STOP", "UNSUPPORTED"):
        assert intent in train_intents


def test_built_splits_no_exact_leakage(built_splits):
    issues = check_leakage(built_splits)
    exact = [i for i in issues if i.startswith("Exact normalized")]
    assert not exact, exact[:5]


def test_manifest_exists_when_built():
    path = SPLITS_DIR / "manifest.json"
    if not path.is_file():
        pytest.skip("manifest missing")
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert "splits" in meta
