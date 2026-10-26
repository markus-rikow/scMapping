#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Keep both passes: the fixed-k run writes the primary k=100 ranking,
# while the dynamic run writes the dynamic-k ranking. The dynamic pass also
# recomputes the fixed-k metrics, but it does not write the k=100 ranking files.
echo "Step 1: Fixed-k ranking (primary k=100, reads stored per-cell CSVs only)"
"$PYTHON_BIN" scripts/05_rank_models.py

echo "Step 2: Dynamic-k ranking (loads every saved model; slow)"
"$PYTHON_BIN" scripts/05_rank_models.py --dynamic

echo "Ranking finished"