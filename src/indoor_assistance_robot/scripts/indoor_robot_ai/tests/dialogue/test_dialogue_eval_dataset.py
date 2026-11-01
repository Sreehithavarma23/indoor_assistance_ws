"""Dialogue eval dataset integrity."""

from src.dialogue.evaluation import load_scenarios


def test_eval_scenarios_cover_key_statuses():
    rows = load_scenarios()
    statuses = {row["expected_status"] for row in rows}
    assert "READY" in statuses
    assert "UNSUPPORTED" in statuses
    assert "CANCELLED" in statuses
    assert len(rows) >= 8
