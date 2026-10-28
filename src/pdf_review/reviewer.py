"""Evidence-grounded reviewer: build input, validate output, render.

Ported unchanged from rerun_general_pdf_generation.py. Note: in the original
script the reviewer's generate_json call reused the SAME closure (and hence
the same retry-hint wording) as extraction's — cli/generate.py preserves this
by passing extraction.EXTRACTION_RETRY_HINT to both calls.
"""
import re

from .normalization import normalize_evidence_ref

ABSENCE_CLAIM = re.compile(
    r"\b(?:the paper|the authors?|the study|the evidence)\s+"
    r"(?:does not|do not|did not|fails? to)\s+"
    r"(?:provide|report|describe|include|state|explain)\b|"
    r"\b(?:not reported|not described|missing from the paper|absent from the paper)\b|"
    r"\b(?:more|additional|further)\s+detail(?:ed|s)?\b|"
    r"\binclude\s+(?:more\s+)?details?\b|"
    r"\bshould\s+specify\b|"
    r"\b(?:lacks?|lacking)\s+(?:detail|description|clarity|specificity)\b|"
    r"\b(?:insufficient(?:ly)?|not\s+(?:sufficiently|adequately|clearly))\s+"
    r"(?:described|explained|detailed|reported)\b|"
    r"\bunclear how\b",
    re.IGNORECASE,
)

REVIEWER_SYSTEM_PROMPT = (
    "You are a rigorous research-paper reviewer. Use only cited structured "
    "evidence and distinguish evidence from inference."
)

REVIEWER_TASK_PROMPT_PREFIX = (
    "Prepare a critical research-paper assessment from the structured evidence below. "
    "Cite exact evidence_ref values. Never claim that information is absent, missing, "
    "insufficiently described, or unreported in the full paper merely because it is not "
    "present in the retrieved evidence. Put coverage uncertainty in unresolved_checks. "
    "A recommendation about a missing item is allowed only when a cited record is explicitly "
    "categorized as limitation. Return JSON only with this schema: "
)


def build_reviewer_input(intermediate_extractions, selected_topics, mode):
    payload = {}
    lookup = {}
    for topic_name in selected_topics:
        payload[topic_name] = []
        for subquery_index, item in enumerate(
            intermediate_extractions[topic_name][mode], 1
        ):
            rows = []
            for record_index, row in enumerate(
                item["record"].get("records", []), 1
            ):
                evidence_ref = f"{topic_name}:{subquery_index}:{record_index}"
                cited_row = dict(row)
                cited_row["evidence_ref"] = evidence_ref
                rows.append(cited_row)
                lookup[evidence_ref] = row
            payload[topic_name].append({
                "subquery": item["subquery"],
                "records": rows,
                "unsupported": item["record"].get("unsupported", []),
            })
    return payload, lookup


def build_reviewer_schema(example_ref):
    """example_ref is None when there is no extracted evidence at all for this mode."""
    if example_ref is not None:
        return {
            "strengths": [{
                "claim": "evidence-grounded strength",
                "evidence_refs": [example_ref],
            }],
            "recommendations": [{
                "area": "review area",
                "observed_basis": "explicit evidence-grounded basis",
                "recommendation": "reviewer recommendation",
                "evidence_refs": [example_ref],
                "inference_label": "reviewer_inference",
                "confidence": "high|medium|low",
            }],
            "unresolved_checks": [{
                "question": "unresolved question",
                "reason": "why the retrieved evidence is insufficient",
                "relevant_evidence_refs": [example_ref],
            }],
        }
    return {
        "strengths": [],
        "recommendations": [],
        "unresolved_checks": [{
            "question": "unresolved question",
            "reason": "why the retrieved evidence is insufficient",
            "relevant_evidence_refs": [],
        }],
    }


def attach_reviewer_pages(value, lookup):
    for item in value["strengths"]:
        item["evidence_pages"] = sorted({
            int(lookup[ref]["pdf_page"]) for ref in item["evidence_refs"]
        })
    for item in value["recommendations"]:
        item["evidence_pages"] = sorted({
            int(lookup[ref]["pdf_page"]) for ref in item["evidence_refs"]
        })
    for item in value["unresolved_checks"]:
        item["relevant_pages"] = sorted({
            int(lookup[ref]["pdf_page"])
            for ref in item.get("relevant_evidence_refs", [])
        })
    return value


def validate_reviewer(value, allowed_refs, evidence_lookup):
    if not isinstance(value, dict):
        raise ValueError("Reviewer output must be a JSON object")
    for key in ("strengths", "recommendations", "unresolved_checks"):
        if not isinstance(value.get(key), list):
            raise ValueError(f"Reviewer field must be a list: {key}")

    def validate_refs(refs, allow_empty=False):
        """Normalize and filter citations; return None if nothing usable remains.

        A single hallucinated or malformed evidence_ref (the model inventing
        "topic:2:2", "topic:2:3" once "topic:2:1" is real, for example) should
        not fail the whole reviewer generation via a deterministic retry loop.
        Unresolvable refs are dropped; the caller drops the whole item only if
        no valid citation remains.
        """
        if not isinstance(refs, list):
            return None if not allow_empty else []
        normalized = [normalize_evidence_ref(ref, allowed_refs) for ref in refs]
        normalized = [ref for ref in normalized if ref in allowed_refs]
        if not normalized and not allow_empty:
            return None
        return normalized

    kept_strengths = []
    for item in value["strengths"]:
        if not isinstance(item.get("claim"), str) or not item["claim"].strip():
            continue  # drop malformed item rather than fail the whole batch
        refs = validate_refs(item.get("evidence_refs"))
        if refs is None:
            continue  # no citation survived; unsupported, so drop it
        item["evidence_refs"] = refs
        kept_strengths.append(item)
    value["strengths"] = kept_strengths

    kept_recommendations = []
    demoted_to_unresolved = []
    for item in value["recommendations"]:
        if any(
            not isinstance(item.get(field), str) or not item[field].strip()
            for field in ("area", "observed_basis", "recommendation")
        ):
            continue  # drop malformed item rather than fail the whole batch
        refs = validate_refs(item.get("evidence_refs"))
        if refs is None:
            continue  # no citation survived; unsupported, so drop it
        item["evidence_refs"] = refs
        item["inference_label"] = "reviewer_inference"
        confidence = item.get("confidence")
        if confidence is None:
            confidence = "medium"
        if isinstance(confidence, str):
            confidence = confidence.strip().lower()
            confidence = {
                "moderate": "medium", "mid": "medium",
                "certain": "high", "very high": "high", "strong": "high",
                "uncertain": "low", "very low": "low", "weak": "low",
            }.get(confidence, confidence)
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"
        item["confidence"] = confidence
        is_absence_claim = ABSENCE_CLAIM.search(item["observed_basis"]) or (
            ABSENCE_CLAIM.search(item["recommendation"])
        )
        has_limitation_evidence = any(
            evidence_lookup[ref].get("category") == "limitation" for ref in refs
        )
        if is_absence_claim and not has_limitation_evidence:
            # Not grounded in an explicit limitation record: this is retrieval-
            # coverage uncertainty, not a stated defect of the paper. Demote to
            # unresolved_checks (the project's own documented intent for this
            # case) instead of failing generation on a deterministic retry loop.
            demoted_to_unresolved.append({
                "question": item["recommendation"],
                "reason": (
                    "Reviewer suggested this as a gap, but no cited record is "
                    "explicitly categorized as a limitation; coverage may be "
                    "incomplete rather than the paper being deficient."
                ),
                "relevant_evidence_refs": refs,
            })
        else:
            kept_recommendations.append(item)
    value["recommendations"] = kept_recommendations

    kept_unresolved = []
    for item in value["unresolved_checks"]:
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            continue
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            continue
        item["relevant_evidence_refs"] = validate_refs(
            item.get("relevant_evidence_refs", []), allow_empty=True
        )
        kept_unresolved.append(item)
    value["unresolved_checks"] = kept_unresolved
    value["unresolved_checks"].extend(demoted_to_unresolved)
    return attach_reviewer_pages(value, evidence_lookup)


def render_reviewer(value):
    lines = ["### Evidence-grounded reviewer assessment", "", "#### Strengths"]
    if value["strengths"]:
        for item in value["strengths"]:
            pages = ", ".join(map(str, item["evidence_pages"]))
            lines.append(f"- {item['claim']} [PDF pages {pages}]")
    else:
        lines.append("- No strength was established from the extracted evidence.")
    lines.extend(["", "#### Recommendations"])
    if value["recommendations"]:
        for item in value["recommendations"]:
            pages = ", ".join(map(str, item["evidence_pages"]))
            lines.append(
                f"- **{item['area']}**: {item['recommendation']} "
                f"Basis: {item['observed_basis']} [PDF pages {pages}]"
            )
    else:
        lines.append("- No recommendation was supported by the extracted evidence.")
    lines.extend(["", "#### Unresolved checks"])
    if value["unresolved_checks"]:
        for item in value["unresolved_checks"]:
            pages = ", ".join(map(str, item["relevant_pages"]))
            citation = f" [PDF pages {pages}]" if pages else ""
            lines.append(f"- {item['question']}: {item['reason']}{citation}")
    else:
        lines.append("- None reported.")
    return "\n".join(lines)
