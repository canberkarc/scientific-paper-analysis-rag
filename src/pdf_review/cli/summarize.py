"""Print a human-readable summary of a strict generation run.

Ported unchanged from run_pipeline.sbatch's inline Python summary block.

Usage:
    python -m pdf_review.cli.summarize --strict-dir /path/to/run/strict
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-dir", type=Path, required=True)
    args = parser.parse_args()
    strict = args.strict_dir

    failure = strict / "generation_failure.json"
    if failure.is_file():
        print("STATUS: generation_failure.json present -- run failed, see", failure)
        return

    audit = json.loads((strict / "audit.json").read_text())
    print("audit status:", audit["status"])
    for key, value in audit["structural_checks"].items():
        print(f"  {key}: {value}")
    flagged = [
        c for c in audit["citation_checks"]
        if c["reported_value_found_in_extracted_page_text"] is False
        or c.get("looks_like_citation_reference_number")
    ]
    print(f"citation_checks: {len(audit['citation_checks'])} total, {len(flagged)} flagged for manual review")
    for c in flagged:
        tag = "citation-number?" if c.get("looks_like_citation_reference_number") else "value-not-found"
        print(f"  [{tag}] {c['topic']}/{c['mode']} p.{c['pdf_page']} value={c['reported_value']!r}")

    comparison = json.loads((strict / "condition_comparison.json").read_text())
    print()
    print("condition_comparison:")
    for mode, stats in comparison["descriptive_summary"].items():
        diag = comparison["page_concentration_diagnostic"][mode]
        share = diag["most_cited_page_share"]
        share_text = f"{share:.0%}" if share is not None else "n/a"
        print(
            f"  {mode}: {stats['extracted_record_count']} records, "
            f"{stats['unsupported_item_count']} unsupported, "
            f"most-cited page {diag['most_cited_page']} ({share_text})"
        )
    print()
    print("Analysis report:", strict / "analysis_report.md")
    print("Worksheets:", strict / "audit_worksheets")


if __name__ == "__main__":
    main()
