"""Build one markdown worksheet per generation mode for the manual semantic
claim audit.

Ported unchanged from build_claim_audit_worksheet.py.
"""


def build_evidence_ref_lookup(intermediate_extractions, mode):
    lookup = {}
    for topic_name, per_mode in intermediate_extractions.items():
        for subquery_index, item in enumerate(per_mode.get(mode, []), 1):
            for record_index, row in enumerate(item["record"].get("records", []), 1):
                ref = f"{topic_name}:{subquery_index}:{record_index}"
                lookup[ref] = {"subquery": item["subquery"], **row}
    return lookup


def page_text_snippet(pages, page_number, max_chars=600):
    for page in pages:
        if page["page_number"] == page_number:
            text = page["text"].strip().replace("\n", " ")
            return text[:max_chars] + ("..." if len(text) > max_chars else "")
    return "(page text not found)"


def render_mode_worksheet(mode, intermediate_extractions, reviewer_assessment, pages):
    lookup = build_evidence_ref_lookup(intermediate_extractions, mode)
    lines = [f"# Claim audit worksheet — {mode}", ""]

    lines.append("## Extraction records (mark [ ] PASS / [ ] FAIL next to each)")
    for ref, row in lookup.items():
        value_part = ""
        if row.get("value") is not None:
            value_part = f" | value: {row['value']} {row.get('unit') or ''}".rstrip()
        lines.append(
            f"- `{ref}` [{row.get('category')}] p.{row['pdf_page']}: "
            f"**{row['claim']}**{value_part}"
        )
        lines.append(f"  - Subquery: {row['subquery']}")
        lines.append(f"  - Page {row['pdf_page']} text: {page_text_snippet(pages, int(row['pdf_page']))}")
        lines.append("  - [ ] Supported as written  [ ] Partially supported  [ ] Not supported")
        lines.append("")

    review = reviewer_assessment.get(mode, {}).get("structured", {})
    lines.append("## Reviewer strengths")
    for item in review.get("strengths", []):
        lines.append(f"- {item['claim']}")
        for ref in item.get("evidence_refs", []):
            source = lookup.get(ref, {})
            lines.append(f"  - `{ref}` -> {source.get('claim', '(unknown ref)')}")
        lines.append("  - [ ] Follows from cited record(s)  [ ] Does not follow")
        lines.append("")

    lines.append("## Reviewer recommendations")
    for item in review.get("recommendations", []):
        lines.append(f"- **{item['area']}**: {item['recommendation']} (basis: {item['observed_basis']})")
        for ref in item.get("evidence_refs", []):
            source = lookup.get(ref, {})
            lines.append(f"  - `{ref}` [{source.get('category')}] -> {source.get('claim', '(unknown ref)')}")
        lines.append("  - [ ] Follows from cited record(s)  [ ] Does not follow")
        lines.append("")

    lines.append("## Unresolved checks (no verification needed, informational)")
    for item in review.get("unresolved_checks", []):
        lines.append(f"- {item['question']} — {item['reason']}")

    return "\n".join(lines)
