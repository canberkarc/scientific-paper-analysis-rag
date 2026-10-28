#!/bin/bash
#SBATCH --job-name=pdf-review-diag
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: sbatch $0 /absolute/path/to/paper.pdf output_directory" >&2
  exit 2
fi

source "$(dirname "${BASH_SOURCE[0]}")/common_env.sh"

python -u -m pdf_review.cli.retrieve \
  --pdf "$1" \
  --output-dir "$2" \
  --topics method \
  --evidence-chunks 50
