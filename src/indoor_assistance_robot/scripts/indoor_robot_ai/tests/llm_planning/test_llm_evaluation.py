"""Offline evaluation smoke test for Phase 7."""

from __future__ import annotations

from src.llm_planning.evaluation import evaluate_default


def test_offline_eval_passes(nlu_engine):
    report = evaluate_default(live=False, engine=nlu_engine)
    assert report["live_provider_used"] is False
    assert report["live_provider_calls"] == 0
    assert report["n_cases"] >= 8
    assert report["case_success_rate"] == 1.0
    assert report["n_structural_match_deterministic"] == report["n_structural_compare"]
