"""Regression tests for extraction.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pdf_review.extraction import split_multi_metric_row, validate_extraction


def test_split_multi_metric_row_distributes_matching_metric_count():
    row = {
        "claim": "x", "value": [0.78, 0.68, 0.61],
        "metric": "AUC, accuracy, F1", "unit": None,
    }
    split = split_multi_metric_row(row)
    assert len(split) == 3
    assert [r["value"] for r in split] == [0.78, 0.68, 0.61]
    assert [r["metric"] for r in split] == ["AUC", "accuracy", "F1"]


def test_split_multi_metric_row_leaves_single_value_untouched():
    row = {"claim": "x", "value": 0.81}
    assert split_multi_metric_row(row) == [row]


def test_split_multi_metric_row_mismatched_metric_count_copies_as_is():
    row = {"claim": "x", "value": [1, 2, 3], "metric": "only one label"}
    split = split_multi_metric_row(row)
    assert len(split) == 3
    assert all(r["metric"] == "only one label" for r in split)


def test_missing_pdf_page_with_single_allowed_page_auto_fills():
    value = {
        "records": [{"claim": "x", "category": "data"}],
        "unsupported": [],
    }
    result = validate_extraction(value, {"3"})
    assert result["records"][0]["pdf_page"] == "3"


def test_missing_pdf_page_with_multiple_allowed_pages_raises_actionable_error():
    """Real failure: this used to raise the confusing 'Unexpected source
    page: .' — now names the claim and the allowed pages explicitly."""
    value = {
        "records": [{"claim": "some claim text", "category": "data"}],
        "unsupported": [],
    }
    with pytest.raises(ValueError, match="missing pdf_page"):
        validate_extraction(value, {"1", "4", "5", "6"})


def test_qualitative_value_like_better_survives_as_null_not_crash():
    value = {
        "records": [{
            "claim": "x", "category": "finding", "pdf_page": "1", "value": "better",
        }],
        "unsupported": [],
    }
    result = validate_extraction(value, {"1"})
    assert result["records"][0]["value"] is None


def test_unrecognized_category_falls_back_to_finding():
    value = {
        "records": [{"claim": "x", "category": "conclusion", "pdf_page": "1"}],
        "unsupported": [],
    }
    result = validate_extraction(value, {"1"})
    assert result["records"][0]["category"] == "finding"
