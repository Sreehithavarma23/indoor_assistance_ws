"""Validation eval harness loads the development JSONL."""

from src.validation.evaluation import load_eval_examples


def test_eval_set_covers_all_statuses():
    rows = load_eval_examples()
    statuses = {row["status"] for row in rows}
    assert statuses == {"VALID", "INCOMPLETE", "AMBIGUOUS", "UNSUPPORTED"}
    assert len(rows) >= 12
    assert all("text" in row and "status" in row for row in rows)
