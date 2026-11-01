"""Offline pipeline evaluation smoke test."""

from __future__ import annotations

from src.pipeline.evaluation import evaluate_default


def test_pipeline_eval_offline(nlu_engine):
    report = evaluate_default(engine=nlu_engine)
    assert report["live_provider_used"] is False
    assert report["n_cases"] >= 12
    assert report["end_to_end_success_rate"] == 1.0
    assert report["n_deterministic_cases"] + report["n_llm_cases"] == report["n_cases"]
