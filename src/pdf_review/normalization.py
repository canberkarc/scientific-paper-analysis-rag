"""Normalize near-miss VLM output into the strict shapes validation expects.

Every function here exists because the model deterministically (greedy
decoding) emits a variant that a generic "please fix it" retry message does
not reliably correct. Ported unchanged from run_general_pdf_review.py.
"""
import re

from .constants import ALLOWED_CATEGORIES, CATEGORY_SYNONYMS

INVALID_NULL_STRINGS = {
    "null", "none", "n/a", "na", "n.a.", "not applicable", "not quantified",
    "unknown", "not specified", "unspecified", "not available", "not reported",
    "not stated", "not given", "not observed", "unclear", "tbd", "-",
    "better", "worse", "higher", "lower", "improved", "increased", "decreased",
    "reduced", "similar", "comparable", "stronger", "weaker", "significant",
    "significantly", "moderate", "positive", "negative", "mixed", "variable",
    "more", "less", "consistent", "inconsistent",
}

NUMERIC_VALUE = re.compile(
    r"^[<>\u2264\u2265~\u2248]?\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?"
    r"(?:\s*(?:\u00b1|\+/-|-|\u2013|to)\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?)?$"
)


def normalize_category(category):
    """Map any category word to the allowed enum; never leaves it unmatched.

    The VLM is shown the exact allowed list in every prompt but still sometimes
    substitutes a synonym (e.g. "conclusion" instead of "finding"). Since
    generation is deterministic (greedy decoding), a validation error alone
    does not reliably make it self-correct on retry. Known variants are mapped
    explicitly; anything still unrecognized falls back to "finding" — the most
    neutral bucket — rather than raising, so a fresh synonym can never crash
    generation. The record itself, its claim and citation, is unaffected; only
    this label is defaulted, and a human reviewing categorized output can
    still see the original wording in the claim text itself.
    """
    if not isinstance(category, str):
        return "finding"
    key = category.strip().lower()
    if key in ALLOWED_CATEGORIES:
        return key
    if key in CATEGORY_SYNONYMS:
        return CATEGORY_SYNONYMS[key]
    if key.endswith("s") and key[:-1] in ALLOWED_CATEGORIES:
        return key[:-1]
    if key.endswith("s") and key[:-1] in CATEGORY_SYNONYMS:
        return CATEGORY_SYNONYMS[key[:-1]]
    return "finding"


def normalize_range_value(value):
    """Convert common non-scalar "value" shapes into the required string form.

    The VLM deterministically emits ranges/bounds as JSON structures instead of
    the required hyphenated string, and a generic retry message does not
    change its output (greedy decoding). Covers:
      [10, 100]        -> "10-100"
      [5.6]             -> "5.6"
      {"min": 1, "max": 9} / {"low": 1, "high": 9} (any 2 numeric-valued keys)
                        -> "1-9"
    Anything else is returned unchanged and still goes through strict
    validation, so genuinely malformed values are still rejected.
    """
    def is_number(item):
        return isinstance(item, (int, float)) and not isinstance(item, bool)

    if isinstance(value, (list, tuple)):
        if len(value) == 1 and is_number(value[0]):
            return f"{value[0]}"
        if len(value) == 2 and all(is_number(item) for item in value):
            return f"{value[0]}-{value[1]}"
        return value
    if isinstance(value, dict) and len(value) == 2:
        numbers = [item for item in value.values() if is_number(item)]
        if len(numbers) == 2:
            return f"{numbers[0]}-{numbers[1]}"
        return value
    return value


def normalize_pdf_page(value, allowed_page_strings):
    """Recover a valid page number from a near-miss pdf_page value.

    If the model writes "2-3", "page 2", "2, 3" or similar instead of a bare
    allowed page number, extract the first embedded integer and use it if it
    is actually one of the allowed pages. Otherwise return the value
    unchanged so the caller's strict check still rejects it.
    """
    text = str(value).strip()
    if text in allowed_page_strings:
        return text
    match = re.search(r"\d+", text)
    if match and match.group() in allowed_page_strings:
        return match.group()
    return text


def normalize_evidence_ref(ref, allowed_refs):
    """Recover a valid evidence_ref from separator/case near-misses.

    Matches e.g. "Findings-1-2" or "findings 1 2" against the canonical
    "findings:1:2" form before rejecting an unrecognized reference.
    """
    if ref in allowed_refs:
        return ref
    if not isinstance(ref, str):
        return ref
    candidate = re.sub(r"[\s\-_]+", ":", ref.strip().lower())
    lowered_map = {allowed.lower(): allowed for allowed in allowed_refs}
    return lowered_map.get(candidate, ref)


def normalize_relevance(value):
    """Coerce a stringified or float relevance score to the required int."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return value


def normalize_extraction(value):
    if isinstance(value, list):
        value = {"records": value, "unsupported": []}
    if not isinstance(value, dict):
        raise ValueError("Extraction response must be a JSON object")
    return value


def normalize_field_list(value):
    """Join a flat list of strings/numbers into a single comma-separated string.

    Optional fields (method, condition, metric, unit) are documented as single
    strings, but the VLM sometimes lists multiple items as a JSON array when a
    claim spans several methods/conditions (e.g. comparing three algorithms at
    once). Joining preserves the content instead of crashing; anything other
    than a flat list of scalars is left unchanged so genuine malformation is
    still rejected.
    """
    if isinstance(value, list) and value and all(
        isinstance(item, (str, int, float)) and not isinstance(item, bool)
        for item in value
    ):
        return ", ".join(str(item) for item in value)
    return value
