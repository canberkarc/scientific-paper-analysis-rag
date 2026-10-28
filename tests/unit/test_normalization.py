"""Regression tests for normalization.py.

Every case here is a real VLM output shape that previously crashed the
pipeline, not a hypothetical. See the project's fix history.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pdf_review.normalization import (
    normalize_category,
    normalize_evidence_ref,
    normalize_extraction,
    normalize_field_list,
    normalize_pdf_page,
    normalize_range_value,
    normalize_relevance,
)


def test_category_synonym_conclusion_maps_to_finding():
    assert normalize_category("conclusion") == "finding"


def test_category_synonym_baseline_maps_to_validation():
    assert normalize_category("baseline") == "validation"


def test_category_synonym_implementation_maps_to_method():
    assert normalize_category("implementation") == "method"


def test_category_unrecognized_word_falls_back_to_finding_never_raises():
    assert normalize_category("some completely novel VLM word") == "finding"


def test_category_plural_of_allowed_word():
    assert normalize_category("methods") == "method"


def test_category_non_string_falls_back_to_finding():
    assert normalize_category(None) == "finding"
    assert normalize_category(42) == "finding"


def test_range_value_two_element_list():
    assert normalize_range_value([10, 100]) == "10-100"


def test_range_value_single_element_list():
    assert normalize_range_value([5.6]) == "5.6"


def test_range_value_min_max_dict():
    assert normalize_range_value({"min": 1, "max": 9}) == "1-9"


def test_range_value_passthrough_for_scalars():
    assert normalize_range_value(0.81) == 0.81
    assert normalize_range_value("0.81") == "0.81"


def test_pdf_page_extracts_embedded_integer():
    assert normalize_pdf_page("page 7", {"7", "8"}) == "7"
    assert normalize_pdf_page("2-3", {"2", "3"}) == "2"


def test_pdf_page_returns_unchanged_when_not_allowed():
    # Caller's strict check must still reject this.
    assert normalize_pdf_page("99", {"1", "2"}) == "99"


def test_evidence_ref_separator_and_case_variants():
    allowed = {"findings:1:2"}
    assert normalize_evidence_ref("Findings-1-2", allowed) == "findings:1:2"
    assert normalize_evidence_ref("findings 1 2", allowed) == "findings:1:2"


def test_relevance_stringified_and_float():
    assert normalize_relevance("2") == 2
    assert normalize_relevance(3.0) == 3
    assert normalize_relevance(True) is True  # bool must pass through untouched


def test_extraction_bare_list_wrapped_into_records_object():
    assert normalize_extraction([{"claim": "x"}]) == {
        "records": [{"claim": "x"}], "unsupported": [],
    }


def test_field_list_multiple_metrics_joined():
    assert normalize_field_list(["accuracy", "precision", "recall"]) == (
        "accuracy, precision, recall"
    )


def test_field_list_non_scalar_list_untouched():
    # A list of dicts is not a valid case for this join; leave it for the
    # caller's strict check to reject.
    value = [{"a": 1}]
    assert normalize_field_list(value) is value
