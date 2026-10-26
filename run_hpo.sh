#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "Step 0: Dry run (prints the sampling/run plan, no training)"
"$PYTHON_BIN" scripts/04_hpo.py --dry-run "$@"

echo "Step 1: HPO (joint scANVI + scArches for all sampled configs)"
"$PYTHON_BIN" scripts/04_hpo.py "$@"

echo "HPO finished"
