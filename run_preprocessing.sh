#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "Step 1: Gene Intersection"
"$PYTHON_BIN" scripts/01_gene_intersection.py

echo "Step 2: Preprocessing Cell Lines"
"$PYTHON_BIN" scripts/02_cell_line_pp.py

echo "Step 3: Preprocessing Tumor"
"$PYTHON_BIN" scripts/03_tumor_pp.py

echo "Validation: Preprocessing artifacts"
"$PYTHON_BIN" scripts/validate_preprocessing.py

echo "Preprocessing finished"
