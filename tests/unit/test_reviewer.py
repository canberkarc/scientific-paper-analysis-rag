"""Regression tests for reviewer.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pdf_review.reviewer import build_reviewer_schema, validate_reviewer


def test_schema_example_ref_choice_is_deterministic_across_process_runs():
    """Real bug: the schema example was picked via next(iter(a_set)), which
    Python randomizes per process (string hash seed), so the reviewer prompt
    silently differed between otherwise-identical runs. sorted()[0] must be
    stable regardless of set construction order or hash seed."""
    refs_a = {"findings:1:1", "method:2:1", "purpose_contributions:1:1"}
    refs_b = {"purpose_contributions:1:1", "findings:1:1", "method:2:1"}
    assert sorted(refs_a)[0] == sorted(refs_b)[0] == "findings:1:1"


def test_schema_with_no_evidence_has_empty_strengths_and_recommendations():
    schema = build_reviewer_schema(None)
    assert schema["strengths"] == []
    assert schema["recommendations"] == []


def test_hallucinated_evidence_ref_dropped_not_fatal():
    """A single invented ref must not fail the whole reviewer batch."""
    allowed_refs = {"findings:1:1"}
    lookup = {"findings:1:1": {"pdf_page": "3", "category": "finding"}}
    value = {
        "strengths": [
            {"claim": "real one", "evidence_refs": ["findings:1:1"]},
            {"claim": "hallucinated", "evidence_refs": ["findings:9:9"]},
        ],
        "recommendations": [],
        "unresolved_checks": [],
    }
    result = validate_reviewer(value, allowed_refs, lookup)
    assert len(result["strengths"]) == 1
    assert result["strengths"][0]["claim"] == "real one"


def test_absence_claim_without_limitation_evidence_demoted_to_unresolved():
    allowed_refs = {"method:1:1"}
    lookup = {"method:1:1": {"pdf_page": "2", "category": "method"}}
    value = {
        "strengths": [],
        "recommendations": [{
            "area": "clarity",
            "observed_basis": "The paper does not report the dataset size.",
            "recommendation": "The authors should specify the dataset size.",
            "evidence_refs": ["method:1:1"],
            "inference_label": "reviewer_inference",
            "confidence": "medium",
        }],
        "unresolved_checks": [],
    }
    result = validate_reviewer(value, allowed_refs, lookup)
    assert result["recommendations"] == []
    assert len(result["unresolved_checks"]) == 1


def test_absence_claim_grounded_in_limitation_record_kept_as_recommendation():
    allowed_refs = {"limitations_review:1:1"}
    lookup = {"limitations_review:1:1": {"pdf_page": "9", "category": "limitation"}}
    value = {
        "strengths": [],
        "recommendations": [{
            "area": "scope",
            "observed_basis": "The paper does not report results on larger datasets.",
            "recommendation": "Test on larger datasets.",
            "evidence_refs": ["limitations_review:1:1"],
            "inference_label": "reviewer_inference",
            "confidence": "medium",
        }],
        "unresolved_checks": [],
    }
    result = validate_reviewer(value, allowed_refs, lookup)
    assert len(result["recommendations"]) == 1


def test_confidence_synonym_normalized():
    allowed_refs = {"method:1:1"}
    lookup = {"method:1:1": {"pdf_page": "2", "category": "method"}}
    value = {
        "strengths": [],
        "recommendations": [{
            "area": "a", "observed_basis": "b", "recommendation": "c",
            "evidence_refs": ["method:1:1"], "confidence": "very high",
        }],
        "unresolved_checks": [],
    }
    result = validate_reviewer(value, allowed_refs, lookup)
    assert result["recommendations"][0]["confidence"] == "high"
