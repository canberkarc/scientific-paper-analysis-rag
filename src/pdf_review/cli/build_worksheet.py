"""Build a single markdown worksheet for the manual semantic claim audit.

CLI wrapper. Ported from build_claim_audit_worksheet.py with identical
behavior. Runs locally, no GPU needed. Reads the strict rerun's output
directory and produces one markdown file per generation mode with every
extraction claim next to its cited page text, and every reviewer strength/
recommendation with its evidence_ref resolved back to the underlying
claim(s) it depends on.

Usage:
    python -m pdf_review.cli.build_worksheet --output-dir /path/to/rerun_output
"""
import argparse
import json
from pathlib import Path

from ..reporting.claim_worksheet import render_mode_worksheet


def load(output_dir, name):
    return json.loads((output_dir / name).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    intermediate_extractions = load(args.output_dir, "intermediate_extractions.json")
    reviewer_assessment = load(args.output_dir, "reviewer_assessment.json")
    pages = load(args.output_dir, "pages.json")

    modes = list(reviewer_assessment.keys())
    worksheets_dir = args.output_dir / "audit_worksheets"
    worksheets_dir.mkdir(exist_ok=True)
    for mode in modes:
        text = render_mode_worksheet(mode, intermediate_extractions, reviewer_assessment, pages)
        path = worksheets_dir / f"{mode}.md"
        path.write_text(text, encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
