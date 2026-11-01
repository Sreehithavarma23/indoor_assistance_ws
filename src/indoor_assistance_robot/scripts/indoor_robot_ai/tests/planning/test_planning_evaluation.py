"""Planning development evaluation smoke tests."""

from __future__ import annotations

from src.planning.evaluation import evaluate_default


def test_eval_dataset_runs():
    report = evaluate_default()
    assert report["n_cases"] >= 8
    assert report["case_success_rate"] == 1.0
    assert report["deterministic_retrieve_repeat"] is True
