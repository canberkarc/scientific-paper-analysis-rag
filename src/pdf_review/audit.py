"""Structural audit: citation checks and descriptive condition comparison.

Ported unchanged from rerun_general_pdf_generation.py. Closures over pages/
selected_topics/intermediate_extractions are now plain functions taking
those as parameters (see cli/generate.py for call sites).
"""
import re
from collections import Counter


def build_normalized_page_text(pages, layout_page_texts):
    return {
        page["page_number"]: " ".join(
            (page["text"] + "\n" + layout_page_texts[page["page_number"] - 1]).casefold().split()
        )
        for page in pages
    }


def looks_like_citation_number(value, unit, page_text):
    """Flag a unitless integer value that only appears in a citation bracket.

    The VLM occasionally copies a bracketed reference marker into the value
    field as if it were a reported quantity: a single ref "[19]", a list
    "[7,8]", or a range "[100-102]". This is informational only (not
    auto-rejected): a genuine small unitless count (e.g. "3 baselines") is
    legitimate, so this only fires when every occurrence of the number on
    the page sits inside an unclosed "[...]" span; a single occurrence
    outside brackets is enough to call it ambiguous and not flag it.
    """
    if unit is not None or value is None:
        return False
    value_text = str(value).strip()
    if not re.fullmatch(r"\d{1,4}", value_text):
        return False
    matches = list(re.finditer(r"\b" + re.escape(value_text) + r"\b", page_text))
    if not matches:
        return False
    for match in matches:
        last_open = page_text.rfind("[", 0, match.start())
        last_close = page_text.rfind("]", 0, match.start())
        if last_open <= last_close:
            return False
    return True


def build_citation_checks(selected_topics, generation_modes, intermediate_extractions, normalized_page_text):
    citation_checks = []
    for topic_name in selected_topics:
        for mode in generation_modes:
            for item in intermediate_extractions[topic_name][mode]:
                for row_index, row in enumerate(
                    item["record"].get("records", []), 1
                ):
                    page_number = int(row["pdf_page"])
                    value = row.get("value")
                    value_found = None
                    if value is not None:
                        value_found = (
                            " ".join(str(value).casefold().split())
                            in normalized_page_text[page_number]
                        )
                    citation_checks.append({
                        "topic": topic_name,
                        "mode": mode,
                        "subquery": item["subquery"],
                        "record_index": row_index,
                        "pdf_page": page_number,
                        "page_is_retrieved": page_number in item["allowed_pages"],
                        "reported_value": value,
                        "reported_value_found_in_extracted_page_text": value_found,
                        "looks_like_citation_reference_number": looks_like_citation_number(
                            value, row.get("unit"), normalized_page_text[page_number]
                        ),
                        "semantic_claim_support_check_required": True,
                    })
    return citation_checks


def build_audit(citation_checks):
    all_pages_valid = all(item["page_is_retrieved"] for item in citation_checks)
    return {
        "status": (
            "STRUCTURAL_PASS_SEMANTIC_REVIEW_REQUIRED"
            if all_pages_valid
            else "STRUCTURAL_FAIL"
        ),
        "structural_checks": {
            "all_citations_use_retrieved_pages": all_pages_valid,
            "strict_json_null_validation": True,
            "strict_numeric_value_validation": True,
            "reviewer_uses_exact_evidence_refs": True,
            "automatic_semantic_claim_verification": False,
            "reused_retrieval_and_visual_evidence": True,
        },
        "citation_checks": citation_checks,
        "limitations": [
            "Semantic support for each generated claim still requires manual review.",
            "A numerical value visible only in page pixels may not occur in extracted page text.",
            "The same Qwen2.5-VL model performs extraction and reviewer generation.",
            "looks_like_citation_reference_number flags a likely bracketed-citation-as-value "
            "confusion (e.g. copying [19] as value=19); it is a heuristic signal for manual "
            "review, not a rejection, since a genuine small unitless count is also possible.",
        ],
    }


def build_page_concentration_diagnostics(selected_topics, generation_modes, intermediate_extractions):
    def pages_for_mode(mode):
        return [
            row.get("pdf_page")
            for topic_name in selected_topics
            for item in intermediate_extractions[topic_name][mode]
            for row in item["record"].get("records", [])
        ]

    page_diagnostics = {}
    for mode in generation_modes:
        pages = pages_for_mode(mode)
        counts = Counter(pages)
        top_page, top_count = (counts.most_common(1)[0] if counts else (None, 0))
        page_diagnostics[mode] = {
            "page_distribution": dict(sorted(counts.items(), key=lambda kv: str(kv[0]))),
            "most_cited_page": top_page,
            "most_cited_page_share": (top_count / len(pages)) if pages else None,
        }
    return page_diagnostics


def build_condition_comparison(
    selected_topics, generation_modes, intermediate_extractions, reviewer_results, page_diagnostics,
):
    return {
        "descriptive_summary": {
            mode: {
                "extracted_record_count": sum(
                    len(item["record"].get("records", []))
                    for topic_name in selected_topics
                    for item in intermediate_extractions[topic_name][mode]
                ),
                "unsupported_item_count": sum(
                    len(item["record"].get("unsupported", []))
                    for topic_name in selected_topics
                    for item in intermediate_extractions[topic_name][mode]
                ),
                "recommendation_count": len(
                    reviewer_results[mode]["structured"]["recommendations"]
                ),
                "unresolved_check_count": len(
                    reviewer_results[mode]["structured"]["unresolved_checks"]
                ),
            }
            for mode in generation_modes
        },
        "page_concentration_diagnostic": page_diagnostics,
        "interpretation_note": (
            "Counts are descriptive and are not quality scores. Manual semantic evidence "
            "review is required before comparing conditions. page_concentration_diagnostic "
            "flags a systemic citation-attribution bias (e.g. extraction defaulting most "
            "records to one page regardless of which page the fact actually came from) "
            "before it requires a full manual worksheet pass; a high most_cited_page_share "
            "(roughly above 0.3-0.4 for a multi-page document) is worth checking first."
        ),
    }
