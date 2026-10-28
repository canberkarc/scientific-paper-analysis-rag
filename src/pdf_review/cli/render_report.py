"""Render the strict pipeline's output into one readable analysis report.

CLI wrapper. Ported from render_analysis_report.py with identical behavior;
the actual rendering logic lives in pdf_review.reporting.analysis_report.

Usage:
    python -m pdf_review.cli.render_report --output-dir general_pdf_review_x_strict
    python -m pdf_review.cli.render_report --output-dir ... --mode hybrid_retrieval_with_images
"""
import argparse
import json
from pathlib import Path

from ..constants import GENERATION_MODES
from ..reporting.analysis_report import render_analysis_report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=list(GENERATION_MODES), default=None,
        help="Which generation mode to render. Default: auto-pick the mode with "
             "the most extracted records relative to unsupported items.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output path for the report. Default: <output-dir>/analysis_report.md",
    )
    args = parser.parse_args()
    for name in ("review_results.json", "audit.json"):
        if not (args.output_dir / name).is_file():
            parser.error(f"{args.output_dir / name} not found")
    if args.out is None:
        args.out = args.output_dir / "analysis_report.md"
    return args


def main():
    args = parse_args()
    review_results = json.loads((args.output_dir / "review_results.json").read_text(encoding="utf-8"))
    audit = json.loads((args.output_dir / "audit.json").read_text(encoding="utf-8"))
    condition_comparison = None
    comparison_path = args.output_dir / "condition_comparison.json"
    if comparison_path.is_file():
        condition_comparison = json.loads(comparison_path.read_text(encoding="utf-8"))

    text = render_analysis_report(review_results, audit, condition_comparison, args.mode)
    args.out.write_text(text, encoding="utf-8")
    print(f"Saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
