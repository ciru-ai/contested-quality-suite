"""IFEval constraint checkers — 25 programmatic instruction-following checks.

Each checker takes a response string and kwargs dict, returns bool.
The instruction IDs follow the format ``category:check_name``.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_CHECKERS: dict[str, Any] = {}


def register(instruction_id: str):
    """Decorator to register a checker function."""

    def decorator(fn):
        _CHECKERS[instruction_id] = fn
        return fn

    return decorator


def check_instruction(
    instruction_id: str,
    response: str,
    kwargs: dict[str, Any],
) -> bool:
    """Run the checker for a given instruction ID.

    Returns ``True`` if the constraint is satisfied, ``False`` otherwise.
    Raises ``KeyError`` if the instruction ID is unknown.
    """
    checker = _CHECKERS.get(instruction_id)
    if checker is None:
        raise KeyError(f"Unknown instruction ID: {instruction_id!r}")
    return checker(response, kwargs)


def available_checkers() -> list[str]:
    """Return all registered instruction IDs."""
    return sorted(_CHECKERS)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _count_words(text: str) -> int:
    return len(text.split())


def _count_sentences(text: str) -> int:
    # Split on sentence-ending punctuation
    sentences = re.split(r"[.!?]+", text)
    return len([s for s in sentences if s.strip()])


def _count_paragraphs(text: str) -> int:
    paragraphs = text.split("\n\n")
    return len([p for p in paragraphs if p.strip()])


def _relation_check(actual: int, expected: int, relation: str) -> bool:
    """Compare actual vs expected using a relation string."""
    rel = relation.lower().strip()
    if rel in ("at least", "atleast"):
        return actual >= expected
    if rel in ("at most", "atmost"):
        return actual <= expected
    if rel in ("exactly", "exact"):
        return actual == expected
    if rel in ("less than",):
        return actual < expected
    if rel in ("more than", "greater than"):
        return actual > expected
    # Default: at least
    return actual >= expected


# ---------------------------------------------------------------------------
# Length constraints
# ---------------------------------------------------------------------------


@register("length_constraints:number_words")
def check_number_words(response: str, kwargs: dict) -> bool:
    num_words = kwargs.get("num_words")
    relation = kwargs.get("relation", "at least")
    if num_words is None:
        return True
    return _relation_check(_count_words(response), num_words, relation)


@register("length_constraints:number_sentences")
def check_number_sentences(response: str, kwargs: dict) -> bool:
    num_sentences = kwargs.get("num_sentences")
    relation = kwargs.get("relation", "at least")
    if num_sentences is None:
        return True
    return _relation_check(_count_sentences(response), num_sentences, relation)


@register("length_constraints:number_paragraphs")
def check_number_paragraphs(response: str, kwargs: dict) -> bool:
    num_paragraphs = kwargs.get("num_paragraphs")
    relation = kwargs.get("relation", "at least")
    if num_paragraphs is None:
        return True
    return _relation_check(_count_paragraphs(response), num_paragraphs, relation)


@register("length_constraints:nth_paragraph_first_word")
def check_nth_paragraph_first_word(response: str, kwargs: dict) -> bool:
    nth = kwargs.get("nth_paragraph")
    first_word = kwargs.get("first_word")
    if nth is None or first_word is None:
        return True
    paragraphs = [p for p in response.split("\n\n") if p.strip()]
    if nth > len(paragraphs) or nth < 1:
        return False
    words = paragraphs[nth - 1].strip().split()
    return bool(words) and words[0].lower() == first_word.lower()


# ---------------------------------------------------------------------------
# Keyword constraints
# ---------------------------------------------------------------------------


@register("keywords:existence")
def check_keywords_existence(response: str, kwargs: dict) -> bool:
    keywords = kwargs.get("keywords")
    if not keywords:
        return True
    lower = response.lower()
    return all(kw.lower() in lower for kw in keywords)


@register("keywords:frequency")
def check_keywords_frequency(response: str, kwargs: dict) -> bool:
    keyword = kwargs.get("keyword")
    frequency = kwargs.get("frequency")
    relation = kwargs.get("relation", "at least")
    if keyword is None or frequency is None:
        return True
    count = response.lower().count(keyword.lower())
    return _relation_check(count, frequency, relation)


@register("keywords:forbidden_words")
def check_forbidden_words(response: str, kwargs: dict) -> bool:
    forbidden = kwargs.get("forbidden_words")
    if not forbidden:
        return True
    lower = response.lower()
    return not any(w.lower() in lower for w in forbidden)


@register("keywords:letter_frequency")
def check_letter_frequency(response: str, kwargs: dict) -> bool:
    letter = kwargs.get("letter")
    let_frequency = kwargs.get("let_frequency")
    let_relation = kwargs.get("let_relation", "at least")
    if letter is None or let_frequency is None:
        return True
    count = response.lower().count(letter.lower())
    return _relation_check(count, let_frequency, let_relation)


# ---------------------------------------------------------------------------
# Format constraints
# ---------------------------------------------------------------------------


@register("detectable_format:number_highlighted_sections")
def check_highlighted_sections(response: str, kwargs: dict) -> bool:
    num = kwargs.get("num_highlights")
    if num is None:
        return True
    # Count *highlighted* sections (markdown bold/italic with *)
    matches = re.findall(r"\*[^*\n]+\*", response)
    return len(matches) >= num


@register("detectable_format:number_bullet_lists")
def check_bullet_lists(response: str, kwargs: dict) -> bool:
    num = kwargs.get("num_bullets")
    if num is None:
        return True
    bullets = re.findall(r"^\s*[-*•]\s+", response, re.MULTILINE)
    return len(bullets) >= num


@register("detectable_format:number_placeholders")
def check_placeholders(response: str, kwargs: dict) -> bool:
    num = kwargs.get("num_placeholders")
    if num is None:
        return True
    placeholders = re.findall(r"\[.+?\]", response)
    return len(placeholders) >= num


@register("detectable_content:number_placeholders")
def check_content_placeholders(response: str, kwargs: dict) -> bool:
    # Same logic as detectable_format version
    return check_placeholders(response, kwargs)


@register("detectable_format:json_format")
def check_json_format(response: str, kwargs: dict) -> bool:
    text = response.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last ``` lines
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines)
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


@register("detectable_format:title")
def check_title(response: str, kwargs: dict) -> bool:
    """Response should have a title — a line at the start that looks like a heading."""
    lines = response.strip().split("\n")
    if not lines:
        return False
    first = lines[0].strip()
    # Markdown heading or a short line without ending period
    if first.startswith("#"):
        return True
    return bool(first) and not first.endswith(".") and len(first.split()) <= 15


@register("detectable_format:multiple_sections")
def check_multiple_sections(response: str, kwargs: dict) -> bool:
    num_sections = kwargs.get("num_sections")
    section_splitter = kwargs.get("section_spliter")  # Note: typo is in the dataset
    if num_sections is None:
        return True
    if section_splitter:
        sections = response.split(section_splitter)
    else:
        # Default: split on markdown headings
        sections = re.split(r"\n#{1,6}\s+", response)
    non_empty = [s for s in sections if s.strip()]
    return len(non_empty) >= num_sections


@register("detectable_format:constrained_response")
def check_constrained_response(response: str, kwargs: dict) -> bool:
    """Response should be very short — one of a few possible answers."""
    # The constraint is that the response is constrained; we just check length
    return len(response.strip().split()) <= 50


# ---------------------------------------------------------------------------
# Punctuation constraints
# ---------------------------------------------------------------------------


@register("punctuation:no_comma")
def check_no_comma(response: str, kwargs: dict) -> bool:
    return "," not in response


# ---------------------------------------------------------------------------
# Start/end constraints
# ---------------------------------------------------------------------------


@register("startend:end_checker")
def check_end_phrase(response: str, kwargs: dict) -> bool:
    end_phrase = kwargs.get("end_phrase")
    if not end_phrase:
        return True
    return response.strip().endswith(end_phrase)


@register("startend:quotation")
def check_quotation(response: str, kwargs: dict) -> bool:
    text = response.strip()
    return (
        (text.startswith('"') and text.endswith('"'))
        or (text.startswith("'") and text.endswith("'"))
        or (text.startswith("\u201c") and text.endswith("\u201d"))
    )


# ---------------------------------------------------------------------------
# Case constraints
# ---------------------------------------------------------------------------


@register("change_case:english_uppercase")
def check_uppercase(response: str, kwargs: dict) -> bool:
    # Only check alphabetic characters
    alpha = "".join(c for c in response if c.isalpha())
    return alpha == alpha.upper() if alpha else True


@register("change_case:english_lowercase")
def check_lowercase(response: str, kwargs: dict) -> bool:
    alpha = "".join(c for c in response if c.isalpha())
    return alpha == alpha.lower() if alpha else True


@register("change_case:english_capital")
def check_capitalize(response: str, kwargs: dict) -> bool:
    """Every word should be capitalized (title case)."""
    words = response.split()
    return all(w[0].isupper() for w in words if w and w[0].isalpha())


# ---------------------------------------------------------------------------
# Combination / misc constraints
# ---------------------------------------------------------------------------


@register("combination:repeat_prompt")
def check_repeat_prompt(response: str, kwargs: dict) -> bool:
    prompt = kwargs.get("prompt_to_repeat")
    if not prompt:
        return True
    return prompt in response


@register("combination:two_responses")
def check_two_responses(response: str, kwargs: dict) -> bool:
    """Response should contain two distinct parts separated by specific markers."""
    # Common separators: "******", "---", or section markers
    separators = ["******", "---", "***"]
    for sep in separators:
        parts = response.split(sep)
        if len(parts) >= 2 and all(p.strip() for p in parts[:2]):
            return True
    return False


@register("language:response_language")
def check_response_language(response: str, kwargs: dict) -> bool:
    """Heuristic check for response language."""
    language = kwargs.get("language")
    if not language:
        return True
    # Simple heuristic: check for language-specific character sets
    lang = language.lower()
    if lang in ("en", "english"):
        # Most characters should be ASCII
        ascii_count = sum(1 for c in response if ord(c) < 128)
        return ascii_count / max(len(response), 1) > 0.8
    if lang in ("zh", "chinese"):
        cjk = sum(1 for c in response if "\u4e00" <= c <= "\u9fff")
        return cjk > 10
    if lang in ("ja", "japanese"):
        jp = sum(1 for c in response if "\u3040" <= c <= "\u309f" or "\u30a0" <= c <= "\u30ff")
        return jp > 5
    if lang in ("ko", "korean"):
        kr = sum(1 for c in response if "\uac00" <= c <= "\ud7a3")
        return kr > 5
    # For other languages, accept by default
    return True


@register("detectable_content:postscript")
def check_postscript(response: str, kwargs: dict) -> bool:
    marker = kwargs.get("postscript_marker", "P.S.")
    return marker in response or "P.S." in response or "PS:" in response
