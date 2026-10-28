"""Regression tests for audit.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pdf_review.audit import build_audit, looks_like_citation_number


def test_bracketed_single_reference_flagged():
    page_text = "prior work has shown this effect [19] under similar conditions"
    assert looks_like_citation_number("19", None, page_text) is True


def test_bracketed_list_reference_flagged():
    page_text = "several studies confirm this [7,8] independently"
    assert looks_like_citation_number("7", None, page_text) is True


def test_genuine_unitless_count_not_flagged_when_also_appears_outside_brackets():
    page_text = "we evaluate 3 baselines across all experiments"
    assert looks_like_citation_number("3", None, page_text) is False


def test_value_with_unit_never_flagged():
    page_text = "the model uses [19] layers"
    assert looks_like_citation_number("19", "layers", page_text) is False


def test_non_numeric_value_never_flagged():
    assert looks_like_citation_number("high", None, "some [19] text") is False


def test_audit_status_pass_when_all_citations_retrieved():
    checks = [{"page_is_retrieved": True}, {"page_is_retrieved": True}]
    audit = build_audit(checks)
    assert audit["status"] == "STRUCTURAL_PASS_SEMANTIC_REVIEW_REQUIRED"
    assert audit["structural_checks"]["automatic_semantic_claim_verification"] is False


def test_audit_status_fail_when_any_citation_not_retrieved():
    checks = [{"page_is_retrieved": True}, {"page_is_retrieved": False}]
    audit = build_audit(checks)
    assert audit["status"] == "STRUCTURAL_FAIL"
