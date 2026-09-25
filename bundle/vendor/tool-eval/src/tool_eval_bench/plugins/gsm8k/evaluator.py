"""GSM8K answer evaluator — extract and compare numeric answers.

Handles multiple answer-extraction strategies:
1. Standard ``#### N`` marker (used by models that follow GSM8K format)
2. Last number in the response (fallback)
3. Normalises commas, dollar signs, percent signs, and whitespace
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(slots=True)
class EvalResult:
    """Result of evaluating a single GSM8K response."""

    correct: bool
    extracted_answer: float | None
    ground_truth: float
    extraction_method: str  # "marker", "last_number", "none"


def extract_answer(text: str) -> tuple[float | None, str]:
    """Extract a numeric answer from the model's response.

    Returns ``(number, method)`` where *method* indicates how the
    answer was found.  Returns ``(None, "none")`` if no number could
    be extracted.
    """
    # Strategy 1: Look for the standard GSM8K ``#### N`` marker
    match = re.search(r"####\s*([^\n]+)", text)
    if match:
        num = _parse_number(match.group(1).strip())
        if num is not None:
            return num, "marker"

    # Strategy 2: Look for "the answer is N" pattern
    match = re.search(
        r"(?:the\s+answer\s+is|answer\s*[:=])\s*\$?\s*([0-9][0-9,]*\.?[0-9]*)",
        text,
        re.IGNORECASE,
    )
    if match:
        num = _parse_number(match.group(1))
        if num is not None:
            return num, "answer_pattern"

    # Strategy 3: Last number in the text (excluding superscripts, dates, etc.)
    # Find all standalone numbers (possibly with commas, decimal points)
    numbers = re.findall(
        r"(?<![a-zA-Z/])(-?\$?\s*[0-9][0-9,]*\.?[0-9]*)(?![a-zA-Z/%])",
        text,
    )
    if numbers:
        # Take the last one
        num = _parse_number(numbers[-1])
        if num is not None:
            return num, "last_number"

    return None, "none"


def _parse_number(s: str) -> float | None:
    """Parse a numeric string, stripping common formatting."""
    s = s.strip()
    # Remove currency symbols, commas, spaces
    s = re.sub(r"[$€£¥,\s]", "", s)
    # Remove trailing period that isn't decimal (e.g. "42.")
    if s.endswith("."):
        s = s[:-1]
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def evaluate_answer(
    model_response: str,
    ground_truth: float,
    *,
    tolerance: float = 1e-5,
) -> EvalResult:
    """Compare the model's extracted answer against the ground truth.

    Uses absolute tolerance for integers and relative tolerance for
    floating-point answers.
    """
    extracted, method = extract_answer(model_response)

    if extracted is None:
        return EvalResult(
            correct=False,
            extracted_answer=None,
            ground_truth=ground_truth,
            extraction_method=method,
        )

    # For integer ground truths, require exact match (after rounding)
    if ground_truth == int(ground_truth):
        correct = math.isclose(extracted, ground_truth, abs_tol=tolerance)
    else:
        # For floats, use relative tolerance
        correct = math.isclose(extracted, ground_truth, rel_tol=1e-3, abs_tol=tolerance)

    return EvalResult(
        correct=correct,
        extracted_answer=extracted,
        ground_truth=ground_truth,
        extraction_method=method,
    )
