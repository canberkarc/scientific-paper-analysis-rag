"""Render the strict pipeline's output into one readable analysis report.

Ported unchanged from render_analysis_report.py. This is the pipeline's own
direct final step — it does NOT wait for or require the manual semantic
audit. It assembles the already-generated, already-strict-validated
extraction and reviewer text (review_results.json, reviewer_assessment.json)
into one document, in topic order, for one generation mode, and appends the
audit's own findings (status, structural checks, flagged citations,
page-concentration diagnostic) as a visible disclaimer rather than hiding it.

This report is NOT a substitute for the manual semantic audit
(claim_worksheet.py) — automatic_semantic_claim_verification is still false.
"""
from ..constants import GENERATION_MODES, TOPICS


def pick_best_mode(condition_comparison, available_modes):
    summary = condition_comparison["descriptive_summary"]
    candidates = [mode for mode in summary if mode in available_modes] or list(available_modes)

    def score(mode):
        stats = summary.get(mode, {})
        return stats.get("extracted_record_count", 0) - 2 * stats.get("unsupported_item_count", 0)

    return max(candidates, key=score)


def format_audit_notes(audit, condition_comparison, mode):
    lines = ["## Audit Notes", ""]
    lines.append(
        "> **Not manually verified.** This report is the automated pipeline's direct "
        "output. Citation pages are format-checked against retrieved evidence, not "
        "semantically confirmed against the source PDF. Verify any claim you plan to "
        "rely on before use — see `claim_worksheet.py` for a systematic "
        "manual check."
    )
    lines.append("")
    lines.append(f"- **Structural status:** `{audit['status']}`")
    for key, value in audit["structural_checks"].items():
        lines.append(f"  - `{key}`: {value}")

    flagged = [
        c for c in audit["citation_checks"]
        if c["mode"] == mode and (
            c["reported_value_found_in_extracted_page_text"] is False
            or c.get("looks_like_citation_reference_number")
        )
    ]
    if flagged:
        lines.append("")
        lines.append(
            f"- **{len(flagged)} citation(s) in this mode flagged for manual review:**"
        )
        for c in flagged:
            tag = (
                "possible citation-number-as-value"
                if c.get("looks_like_citation_reference_number")
                else "value not found in extracted page text"
            )
            lines.append(
                f"  - [{c['topic']}] p.{c['pdf_page']} value={c['reported_value']!r} — {tag}"
            )

    if condition_comparison:
        diag = condition_comparison.get("page_concentration_diagnostic", {}).get(mode)
        if diag and diag.get("most_cited_page_share") is not None:
            share = diag["most_cited_page_share"]
            lines.append("")
            lines.append(
                f"- **Page concentration:** page {diag['most_cited_page']} accounts for "
                f"{share:.0%} of citations in this mode"
                + (" — worth checking for a systemic attribution bias." if share > 0.35 else ".")
            )

    for limitation in audit.get("limitations", []):
        lines.append(f"- {limitation}")
    return "\n".join(lines)


def render_analysis_report(review_results, audit, condition_comparison=None, mode=None):
    """Build the report text. Raises ValueError on schema/mode problems,
    exactly as the original script's argument validation did.
    """
    reviewer_assessment = review_results.get("reviewer_assessment", {})
    results_by_topic = {item["topic"]: item for item in review_results["results"]}
    available_modes = {
        m
        for item in results_by_topic.values()
        for m, entry in item.get("answers", {}).items()
        if entry.get("answer")
    }
    if not available_modes:
        raise ValueError("review_results.json has no non-empty answers for any mode")

    if mode is None:
        mode = (
            pick_best_mode(condition_comparison, available_modes) if condition_comparison
            else sorted(available_modes)[0]
        )
    elif mode not in available_modes:
        raise ValueError(
            f"Mode {mode!r} has no content in review_results.json. "
            f"Modes with content: {sorted(available_modes)}"
        )

    sections = [
        "# Automated Analysis Report",
        "",
        f"*Generation mode: `{mode}`*",
        "",
        "> **Not manually verified.** See Audit Notes at the end of this report "
        "before treating any specific claim, number, or page citation as confirmed.",
        "",
    ]
    for topic_name in TOPICS:
        item = results_by_topic.get(topic_name)
        if item is None:
            continue
        answer = item["answers"].get(mode, {}).get("answer")
        if answer:
            sections.append(answer)
            sections.append("")

    review_text = reviewer_assessment.get(mode, {}).get("review")
    if review_text:
        sections.append(review_text)
        sections.append("")

    sections.append(format_audit_notes(audit, condition_comparison, mode))
    return "\n".join(sections)
