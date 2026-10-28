#!/bin/bash
# Shared environment setup for every pdf_review Slurm job.
# Source this, don't execute it: `source scripts/slurm/common_env.sh`
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -n "${VENV_PATH:-}" ]]; then
    source "$VENV_PATH/bin/activate"
fi
cd "$REPO_ROOT"
mkdir -p logs

# Makes `python -m pdf_review.cli.*` work without requiring `pip install -e .`
# first; installing the package is still recommended (see README), this is
# just a safety net so a fresh checkout works immediately.
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
