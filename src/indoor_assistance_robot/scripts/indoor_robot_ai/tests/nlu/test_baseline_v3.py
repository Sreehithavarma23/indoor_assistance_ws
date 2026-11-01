"""Regression checks for the explicit, provenance-recorded NLU baseline."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.config_loader import PROJECT_ROOT, load_nlu
from src.nlu.baseline_dataset import DATASET_DIR, check_split_leakage, sha256_file
from src.nlu.exceptions import IntentModelError
from src.nlu.intent_classifier import TfidfLogRegIntentClassifier
from src.nlu.nlu import NLUEngine
from src.nlu.train_baseline_v3 import DEFAULT_ARTIFACT, train_and_save
from src.validation import validate
from src.validation.types import ValidationStatus


def test_inference_uses_explicit_configured_v3_artifact():
    configured = Path(load_nlu()["tfidf_model_path"])
    if not configured.is_absolute():
        configured = PROJECT_ROOT / configured
    assert configured == DEFAULT_ARTIFACT
    engine = NLUEngine.load()
    assert engine.classifier.training_metadata["model_name"] == "tfidf_logistic_regression_baseline_v3"


def test_missing_artifact_does_not_trigger_implicit_training(tmp_path, monkeypatch):
    def unexpected_train(cls):
        raise AssertionError("implicit training was called")

    monkeypatch.setattr(NLUEngine, "train_default", classmethod(unexpected_train))
    with pytest.raises(IntentModelError, match="Intent model not found"):
        NLUEngine.load_or_train(tmp_path / "missing.joblib")


def test_training_and_inference_share_outer_whitespace_preprocessing():
    engine = NLUEngine.load()
    classifier = engine.classifier
    plain = classifier.pipeline.predict_proba(["go to the kitchen"])[0]
    padded = classifier.pipeline.predict_proba(["  go to the kitchen  "])[0]
    np.testing.assert_allclose(plain, padded, rtol=0.0, atol=0.0)
    assert classifier.predict("  go to the kitchen  ").intent == classifier.predict(
        "go to the kitchen"
    ).intent


def test_split_leakage_checker_flags_available_family_and_template_metadata():
    report = check_split_leakage(
        {
            "train": [
                {
                    "text": "go to kitchen",
                    "source_id": "session-a",
                    "template_id": "nav-1",
                    "paraphrase_family_id": "fam-1",
                }
            ],
            "test": [
                {
                    "text": "please go to kitchen",
                    "source_id": "session-b",
                    "template_id": "nav-1",
                    "paraphrase_family_id": "fam-1",
                }
            ],
        }
    )
    kinds = {issue["kind"] for issue in report["issues"]}
    assert "template_id_overlap" in kinds
    assert "paraphrase_family_id_overlap" in kinds
    assert report["independence_verifiable"] is True


def test_split_leakage_checker_marks_missing_metadata_unverifiable():
    report = check_split_leakage(
        {"train": [{"text": "go"}], "test": [{"text": "stop"}]}
    )
    assert report["independence_verifiable"] is False
    assert report["coverage"]["template_id"]["present_rows"] == 0


def test_dataset_and_artifact_provenance_hashes_are_recorded():
    manifest = json.loads((DATASET_DIR / "manifest.json").read_text())
    assert manifest["train"]["rows"] == 374
    assert manifest["train"]["sha256"] == sha256_file(DATASET_DIR / "train.jsonl")
    assert manifest["of_which_were_in_v2_train"] == 42
    assert manifest["excluded_legacy_test_rows"] == 70
    assert all(
        row.get("source_id") is None and row.get("template_id") is None
        for row in json.loads((DATASET_DIR / "legacy_test_exclusion.json").read_text())["rows"]
    )

    provenance_path = DEFAULT_ARTIFACT.with_suffix(
        DEFAULT_ARTIFACT.suffix + ".provenance.json"
    )
    provenance = json.loads(provenance_path.read_text())
    assert provenance["dataset_sha256"] == sha256_file(DATASET_DIR / "train.jsonl")
    assert provenance["artifact_sha256"] == sha256_file(DEFAULT_ARTIFACT)
    assert provenance["random_seed"] == 42
    NLUEngine.load()  # verifies the artifact against its SHA-256 sidecar


def test_model_load_rejects_artifact_hash_mismatch(tmp_path):
    copied = tmp_path / "model.joblib"
    sidecar = copied.with_suffix(copied.suffix + ".provenance.json")
    shutil.copyfile(DEFAULT_ARTIFACT, copied)
    source_sidecar = DEFAULT_ARTIFACT.with_suffix(DEFAULT_ARTIFACT.suffix + ".provenance.json")
    shutil.copyfile(source_sidecar, sidecar)
    copied.write_bytes(copied.read_bytes() + b"tamper")
    with pytest.raises(IntentModelError, match="SHA-256 mismatch"):
        TfidfLogRegIntentClassifier.load(copied)


def test_training_is_deterministic_for_same_rows_seed_and_environment(tmp_path):
    train_path = DATASET_DIR / "train.jsonl"
    first = train_and_save(train_path, tmp_path / "one.joblib", seed=42)
    second = train_and_save(train_path, tmp_path / "two.joblib", seed=42)
    assert first["artifact_sha256"] == second["artifact_sha256"]


def test_oos_command_and_invalid_catalog_task_are_rejected_by_exact_artifact():
    engine = NLUEngine.load()
    oos = validate(engine.parse("Do a backflip."))
    assert oos.intent == "UNSUPPORTED"
    assert oos.status == ValidationStatus.UNSUPPORTED

    invalid_task = validate(engine.parse("Go to the rooftop."))
    assert invalid_task.status == ValidationStatus.UNSUPPORTED
