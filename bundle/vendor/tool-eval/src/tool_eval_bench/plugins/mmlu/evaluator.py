"""MMLU answer evaluator — extract and compare multiple-choice answers."""

from __future__ import annotations

import re
from dataclasses import dataclass

_VALID_LETTERS = {"A", "B", "C", "D"}

# Patterns ordered by priority
_ANSWER_IS_RE = re.compile(
    r"(?:the\s+)?answer\s+is\s*:?\s*\(?([A-D])\)?",
    re.IGNORECASE,
)
_ANSWER_COLON_RE = re.compile(r"answer\s*:\s*\(?([A-D])\)?", re.IGNORECASE)
_STANDALONE_LETTER_RE = re.compile(r"\b([A-D])\b")


@dataclass(slots=True)
class MMLUEvalResult:
    """Result of evaluating a single MMLU question."""

    correct: bool
    extracted_answer: str | None  # "A", "B", "C", or "D"
    ground_truth_letter: str  # "A", "B", "C", or "D"
    ground_truth_index: int  # 0-3
    extraction_method: str  # "exact", "answer_pattern", "first_letter", "none"


def extract_answer(response: str) -> tuple[str | None, str]:
    """Extract a multiple-choice letter from a model response.

    Returns ``(letter, method)`` where *letter* is A/B/C/D or ``None``,
    and *method* describes how it was found.
    """
    text = response.strip()
    if not text:
        return None, "none"

    # 1. Exact single letter (possibly with period/parenthesis)
    cleaned = text.strip(".()")
    if cleaned.upper() in _VALID_LETTERS and len(cleaned) == 1:
        return cleaned.upper(), "exact"

    # 2. "The answer is B" / "Answer: C"
    m = _ANSWER_IS_RE.search(text)
    if m:
        return m.group(1).upper(), "answer_pattern"

    m = _ANSWER_COLON_RE.search(text)
    if m:
        return m.group(1).upper(), "answer_pattern"

    # 3. First standalone A/B/C/D letter in the response
    m = _STANDALONE_LETTER_RE.search(text)
    if m:
        return m.group(1).upper(), "first_letter"

    return None, "none"


def evaluate_answer(response: str, ground_truth: int) -> MMLUEvalResult:
    """Evaluate a model response against the ground truth.

    Parameters
    ----------
    response
        The model's raw text response.
    ground_truth
        The correct answer index (0=A, 1=B, 2=C, 3=D).
    """
    gt_letter = "ABCD"[ground_truth]
    extracted, method = extract_answer(response)

    return MMLUEvalResult(
        correct=extracted == gt_letter,
        extracted_answer=extracted,
        ground_truth_letter=gt_letter,
        ground_truth_index=ground_truth,
        extraction_method=method,
    )
