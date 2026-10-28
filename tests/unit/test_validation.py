"""Regression tests for validation.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pdf_review.validation import validate_optional_extraction_fields, validate_reused_evidence


def test_qualitative_comparison_word_in_value_is_auto_nulled():
    """Real failure: value="better" crashed generation four attempts in a row
    before this was made an auto-null instead of a hard reject."""
    row = {"claim": "x", "value": "better", "unit": None}
    validate_optional_extraction_fields(row)
    assert row["value"] is None
    assert row["unit"] is None


def test_genuine_numeric_value_untouched():
    row = {"claim": "x", "value": 0.81, "unit": None}
    validate_optional_extraction_fields(row)
    assert row["value"] == 0.81


def test_unrecognized_qualitative_text_still_rejected():
    """The safety net: a word not in INVALID_NULL_STRINGS must still raise,
    not silently pass through as a fabricated numeric value."""
    row = {"claim": "x", "value": "excellent performance", "unit": None}
    with pytest.raises(ValueError):
        validate_optional_extraction_fields(row)


def test_empty_string_unit_becomes_null_instead_of_raising():
    row = {"claim": "x", "value": 93, "unit": ""}
    validate_optional_extraction_fields(row)
    assert row["unit"] is None


def test_value_null_forces_unit_null_too():
    row = {"claim": "x", "value": None, "unit": "subjects"}
    validate_optional_extraction_fields(row)
    assert row["unit"] is None


def test_range_value_string_with_unicode_dash_accepted():
    row = {"claim": "x", "value": "10\u201320", "unit": None}
    validate_optional_extraction_fields(row)  # must not raise


def test_reused_evidence_topic_mismatch_raises():
    selected_topics = {"method": {"subqueries": ["q1"]}}
    with pytest.raises(ValueError):
        validate_reused_evidence({}, selected_topics, page_count=5)


def test_reused_evidence_well_formed_passes():
    selected_topics = {"method": {"subqueries": ["q1"]}}
    evidence = {
        "method": {
            "subquery_traces": [{
                "subquery": "q1",
                "selected_text": [],
                "selected_text_pages": [1],
                "selected_image_pages": [],
            }]
        }
    }
    validate_reused_evidence(evidence, selected_topics, page_count=5)  # must not raise
