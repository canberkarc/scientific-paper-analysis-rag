#!/bin/bash
#SBATCH --job-name=pdf-ask-question
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: sbatch $0 evidence_dir output_dir \"question text\" [--with-images]" >&2
  exit 2
fi

EVIDENCE_DIR="$1"
OUTPUT_DIR="$2"
QUESTION="$3"
shift 3

source "$(dirname "${BASH_SOURCE[0]}")/common_env.sh"

python -u -m pdf_review.cli.ask \
  --evidence-dir "$EVIDENCE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --question "$QUESTION" \
  "$@"
