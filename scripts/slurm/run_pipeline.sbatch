#!/bin/bash
#SBATCH --job-name=pdf-review-pipeline
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch $0 /path/to/paper.pdf output_base_dir [\"optional ad-hoc question\"]" >&2
  echo "  output_base_dir must not already exist." >&2
  exit 2
fi

PDF="$1"
BASE="$2"
QUESTION="${3:-}"

if [[ -e "$BASE" ]]; then
  echo "Output base directory already exists: $BASE" >&2
  exit 2
fi

source "$(dirname "${BASH_SOURCE[0]}")/common_env.sh"

EVIDENCE_DIR="$BASE/evidence"
STRICT_DIR="$BASE/strict"

echo "=== Step 1/5: retrieval + visual page scoring ==="
python -u -m pdf_review.cli.retrieve \
  --pdf "$PDF" \
  --output-dir "$EVIDENCE_DIR" \
  --evidence-chunks 8 \
  --max-chunks-per-page 2

echo "=== Step 2/5: strict extraction, reviewer, audit ==="
python -u -m pdf_review.cli.generate \
  --pdf "$PDF" \
  --evidence "$EVIDENCE_DIR/evidence.json" \
  --output-dir "$STRICT_DIR" \
  --max-new-tokens 1536

echo "=== Step 3/5: claim audit worksheets ==="
python -u -m pdf_review.cli.build_worksheet --output-dir "$STRICT_DIR"

echo "=== Step 4/5: analysis report ==="
python -u -m pdf_review.cli.render_report --output-dir "$STRICT_DIR"

if [[ -n "$QUESTION" ]]; then
  echo "=== Step 5/5: ad-hoc question ==="
  python -u -m pdf_review.cli.ask \
    --evidence-dir "$EVIDENCE_DIR" \
    --output-dir "$BASE/question" \
    --question "$QUESTION"
else
  echo "=== Step 5/5: skipped (no ad-hoc question given) ==="
fi

echo "=== Summary ==="
python -u -m pdf_review.cli.summarize --strict-dir "$STRICT_DIR"

echo "=== Done: $BASE ==="
