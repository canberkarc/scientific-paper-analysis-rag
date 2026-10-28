"""Integration test: reporting pipeline against a real-shaped fixture.

No GPU needed — exercises the exact contract render_analysis_report.py and
build_claim_audit_worksheet.py depend on (review_results.json, audit.json,
condition_comparison.json, intermediate_extractions.json,
reviewer_assessment.json, pages.json).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pdf_review.reporting.analysis_report import render_analysis_report
from pdf_review.reporting.claim_worksheet import render_mode_worksheet

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "sample_strict_output"


def _load(name):
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_analysis_report_includes_not_verified_disclaimer_and_flagged_citation():
    review_results = _load("review_results.json")
    audit = _load("audit.json")
    condition_comparison = _load("condition_comparison.json")

    text = render_analysis_report(review_results, audit, condition_comparison)

    assert "Not manually verified" in text
    assert "Purpose Contributions" in text
    assert "value not found in extracted page text" in text
    assert "STRUCTURAL_PASS_SEMANTIC_REVIEW_REQUIRED" in text


def test_analysis_report_auto_picks_best_scoring_mode():
    review_results = _load("review_results.json")
    audit = _load("audit.json")
    condition_comparison = _load("condition_comparison.json")

    text = render_analysis_report(review_results, audit, condition_comparison)
    assert "hybrid_retrieval_with_images" in text


def test_claim_worksheet_resolves_evidence_ref_to_source_claim():
    intermediate_extractions = _load("intermediate_extractions.json")
    reviewer_assessment = _load("reviewer_assessment.json")
    pages = _load("pages.json")

    worksheet = render_mode_worksheet(
        "hybrid_retrieval_with_images", intermediate_extractions, reviewer_assessment, pages
    )
    assert "purpose_contributions:1:1" in worksheet
    assert "The study investigates PD classification." in worksheet
    assert "Supported as written" in worksheet
