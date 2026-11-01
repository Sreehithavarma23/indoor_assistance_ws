"""Evaluation harness uses the real train/test files."""

from src.nlu.evaluation import evaluate_engine


def test_evaluation_runs_on_held_out_split(nlu_engine, test_examples):
    report = evaluate_engine(nlu_engine, test_examples)
    assert report["n_examples"] == len(test_examples)
    assert 0.0 <= report["intent_accuracy"] <= 1.0
    assert "micro" in report["entity"]
    assert report["errors"] is not None
    assert "not a real-world" in report["note"]
