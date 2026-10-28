#!/bin/bash
#SBATCH --job-name=pdf-review-strict
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: sbatch $0 /absolute/path/to/paper.pdf /absolute/path/to/evidence.json output_directory" >&2
  exit 2
fi

source "$(dirname "${BASH_SOURCE[0]}")/common_env.sh"

python -u -m pdf_review.cli.generate \
  --pdf "$1" \
  --evidence "$2" \
  --output-dir "$3" \
  --json-retries 3 \
  --max-new-tokens 1536 \
  --reviewer-max-new-tokens 1024 \
  --max-pixels 1003520
