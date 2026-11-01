"""WER and CER. Implemented here so evaluation does not need extra libraries."""

from __future__ import annotations

import string
from typing import List, Sequence


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
) -> str:
    """Normalize a transcript before alignment."""
    cleaned = text.strip()
    if lowercase:
        cleaned = cleaned.lower()
    if remove_punctuation:
        translator = str.maketrans("", "", string.punctuation)
        cleaned = cleaned.translate(translator)
    return " ".join(cleaned.split())


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Levenshtein distance over tokens or characters."""
    ref = list(reference)
    hyp = list(hypothesis)
    rows = len(ref) + 1
    cols = len(hyp) + 1
    previous = list(range(cols))
    for i in range(1, rows):
        current = [i] + [0] * (cols - 1)
        for j in range(1, cols):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Insertions + deletions + substitutions, divided by reference word count."""
    ref_words: List[str] = reference.split()
    hyp_words: List[str] = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over characters, including spaces, / |reference|."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(list(reference), list(hypothesis)) / len(reference)
