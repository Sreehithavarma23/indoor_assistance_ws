"""Evaluation-harness tests. They do not run Faster-Whisper or invent scores."""

from pathlib import Path

import pytest

from src.speech.evaluate import critical_term_error_rate, read_manifest
from src.speech.metrics import word_error_rate


def test_read_manifest_skips_comments_and_blank_lines(tmp_path: Path):
    path = tmp_path / "manifest.csv"
    path.write_text(
        "audio_path,reference\n"
        "# data/raw_audio/skip_me.wav,should be ignored\n"
        "\n"
        "data/raw_audio/one.wav,go to the kitchen\n",
        encoding="utf-8",
    )
    rows = read_manifest(path)
    assert rows == [
        {
            "audio_path": "data/raw_audio/one.wav",
            "reference": "go to the kitchen",
            "critical_terms": "",
        }
    ]


def test_read_manifest_accepts_critical_terms(tmp_path: Path):
    path = tmp_path / "manifest.csv"
    path.write_text(
        "audio_path,reference,critical_terms\n"
        "data/raw_audio/one.wav,bring the bottle,bottle;not\n",
        encoding="utf-8",
    )
    rows = read_manifest(path)
    assert rows[0]["critical_terms"] == "bottle;not"


def test_read_manifest_missing_file():
    with pytest.raises(FileNotFoundError):
        read_manifest("does_not_exist.csv")


def test_read_manifest_requires_columns(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text("file,text\nfoo.wav,hello\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_manifest(path)


def test_header_only_manifest_is_empty(tmp_path: Path):
    path = tmp_path / "empty.csv"
    path.write_text("audio_path,reference\n", encoding="utf-8")
    assert read_manifest(path) == []


def test_wer_used_by_evaluation_is_deterministic():
    # Guardrail: the report uses this function; it must not invent a score.
    assert word_error_rate("stop", "stop") == 0.0


def test_critical_term_error_rate_counts_missing_terms():
    score = critical_term_error_rate(
        "do not bring the bottle",
        "bring the bottle",
        ["not", "bottle"],
    )
    assert score == 0.5


def test_evaluate_manifest_skips_missing_audio_and_does_not_invent_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from src.speech.evaluate import evaluate_manifest

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "audio_path,reference\nmissing.wav,bring me the bottle\n",
        encoding="utf-8",
    )

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("ASR must not run for a missing audio file")

    monkeypatch.setattr("src.speech.evaluate.transcribe_file", should_not_run)
    summary = evaluate_manifest(manifest)
    assert summary["n_items"] == 0
    assert summary["n_skipped"] == 1
    assert summary["mean_wer"] is None
    assert summary["mean_cer"] is None
