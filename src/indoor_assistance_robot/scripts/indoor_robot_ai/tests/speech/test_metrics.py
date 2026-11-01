"""Unit tests for WER/CER. These check the metric math, not ASR quality."""

from src.speech.metrics import (
    character_error_rate,
    normalize_text,
    word_error_rate,
)


def test_normalize_text_lowercases_and_strips_punctuation():
    assert normalize_text("Go to the Kitchen!") == "go to the kitchen"


def test_wer_is_zero_for_identical_sentences():
    assert word_error_rate("bring the bottle", "bring the bottle") == 0.0


def test_wer_counts_one_substitution():
    assert word_error_rate("a b c", "a b d") == 1 / 3


def test_cer_is_zero_for_identical_strings():
    assert character_error_rate("abc", "abc") == 0.0


def test_cer_counts_one_insertion():
    assert character_error_rate("ab", "abc") == 0.5


def test_empty_reference_and_empty_hypothesis():
    assert word_error_rate("", "") == 0.0
    assert character_error_rate("", "") == 0.0


def test_empty_reference_with_hypothesis_is_one():
    assert word_error_rate("", "hello") == 1.0
    assert character_error_rate("", "x") == 1.0
