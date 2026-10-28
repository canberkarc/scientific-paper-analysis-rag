"""Regression tests for io.py's parse_json_response repair logic."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pdf_review.io import parse_json_response


def test_code_fence_stripped():
    text = '```json\n{"a": 1}\n```'
    assert parse_json_response(text) == {"a": 1}


def test_trailing_comma_before_closing_brace_repaired():
    text = '{"a": 1, "b": 2,}'
    assert parse_json_response(text) == {"a": 1, "b": 2}


def test_trailing_comma_before_closing_bracket_repaired():
    text = '{"a": [1, 2,]}'
    assert parse_json_response(text) == {"a": [1, 2]}


def test_multiple_bare_numbers_after_colon_wrapped_into_array():
    """Real failure: value: 0.78, 86.1, 0.61 (three metrics crammed into one
    field with no array brackets) previously caused an opaque parse failure."""
    text = '{"value": 0.78, 86.1, 0.61}'
    result = parse_json_response(text)
    assert result["value"] == [0.78, 86.1, 0.61]


def test_stray_trailing_quote_tolerated():
    text = '{"a": 1}"'
    assert parse_json_response(text) == {"a": 1}


def test_genuine_truncation_still_raises():
    """A real mid-token-limit cutoff must not be silently accepted."""
    text = '{"records": [{"claim": "x", "method'
    with pytest.raises(ValueError):
        parse_json_response(text)


def test_text_after_json_value_raises():
    text = '{"a": 1} some extra prose the model added'
    with pytest.raises(ValueError):
        parse_json_response(text)


def test_no_json_object_or_array_raises():
    with pytest.raises(ValueError):
        parse_json_response("just plain text, no JSON at all")
