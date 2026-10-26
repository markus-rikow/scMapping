"""Validate and summarize the artifacts produced by preprocessing steps 01-03."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config


SUMMARY_OBS_COLUMNS = [
    "dataset",
    "batch",
    "chemistry",
    "cell_line",
    "donor_id",
    "label",
    "cell_type",
    "mmr_status",
    "pool_id",
]

CELL_LINE_REQUIRED_OBS = [
    "cell_id",
    "dataset",
    "batch",
    "chemistry",
    "cell_line",
    "donor_id",
    "label",
    "mmr_status",
    "pool_id",
]

TUMOR_REQUIRED_OBS = [
    "cell_id",
    "dataset",
    "batch",
    "chemistry",
    "cell_line",
    "donor_id",
    "label",
    "mmr_status",
    "biosample_id",
    "cell_type",
]


def value_counts(series):
    return {
        str(key): int(value)
        for key, value in series.value_counts(dropna=False).items()
    }


def matrix_checks(adata):
    nonnegative = True
    finite = True
    integer_valued = True
    matrix_type = type(adata.X).__name__

    for matrix, _, _ in adata.chunked_X(10000):
        values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
        if values.size == 0:
            continue
        finite &= bool(np.isfinite(values).all())
        nonnegative &= bool((values >= 0).all())
        integer_valued &= bool(np.equal(values, np.floor(values)).all())

    return {
        "dtype": str(adata.X.dtype),
        "type": matrix_type,
        "sparse": bool(sparse.issparse(adata.X) or "Sparse" in matrix_type),
        "finite": finite,
        "nonnegative": nonnegative,
        "integer_valued": integer_valued,
    }


def summarize_adata(path, required_obs_columns):
    adata = ad.read_h5ad(path, backed="r")
    try:
        obs = adata.obs
        var = adata.var
        summary = {
            "path": str(path),
            "shape": {"cells": int(adata.n_obs), "genes": int(adata.n_vars)},
            "obs_columns": list(obs.columns),
            "var_columns": list(var.columns),
            "obs_names_unique": bool(adata.obs_names.is_unique),
            "var_names_unique": bool(adata.var_names.is_unique),
            "missing_required_obs_columns": sorted(
                set(required_obs_columns) - set(obs.columns)
            ),
            "obs_missing_values": {
                column: int(obs[column].isna().sum()) for column in obs.columns
            },
            "var_missing_values": {
                column: int(var[column].isna().sum()) for column in var.columns
            },
            "matrix": matrix_checks(adata),
            "value_counts": {
                column: value_counts(obs[column])
                for column in SUMMARY_OBS_COLUMNS
                if column in obs.columns
            },
        }

        if "gene_id" in var.columns:
            summary["gene_id_unique"] = bool(var["gene_id"].is_unique)
        if "cell_line" in obs.columns:
            summary["cell_counts"] = value_counts(obs["cell_line"])

        return summary
    finally:
        if adata.isbacked:
            adata.file.close()


def validate_summary(summary, name, errors):
    if not summary["obs_names_unique"]:
        errors.append(f"{name}: observation names are not unique")
    if not summary["var_names_unique"]:
        errors.append(f"{name}: variable names are not unique")
    if summary["missing_required_obs_columns"]:
        errors.append(
            f"{name}: missing required obs columns "
            f"{summary['missing_required_obs_columns']}"
        )
    matrix = summary["matrix"]
    for property_name in ("finite", "nonnegative", "integer_valued"):
        if not matrix[property_name]:
            errors.append(f"{name}: matrix failed {property_name} count validation")
    if summary.get("gene_id_unique") is False:
        errors.append(f"{name}: gene_id values are not unique")


def main():
    config = load_config()
    processed = config["paths"]["processed"]
    gene_path = Path(processed["genes_combined"])
    cell_line_path = Path(processed["pp_cell_line"])
    tumor_path = Path(processed["pp_tumor"])
    report_path = Path(processed["preprocessing_report"])

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "error",
        "genes": {},
        "cell_line": {},
        "tumor": {},
        "cross_dataset": {},
        "warnings": [],
        "errors": [],
    }

    try:
        genes = pd.read_csv(gene_path)
        report["genes"] = {
            "path": str(gene_path),
            "rows": int(len(genes)),
            "columns": list(genes.columns),
            "gene_id_unique": bool(genes["gene_id"].is_unique),
            "gene_symbol_unique": bool(genes["gene_symbol"].is_unique),
            "missing_gene_ids": int(genes["gene_id"].isna().sum()),
            "missing_gene_symbols": int(genes["gene_symbol"].isna().sum()),
        }
        if not report["genes"]["gene_id_unique"]:
            report["errors"].append("genes: gene_id values are not unique")
        if not report["genes"]["gene_symbol_unique"]:
            report["warnings"].append("genes: gene_symbol values are not unique")
    except Exception as exc:
        report["errors"].append(f"genes: could not read or validate file: {exc}")

    try:
        report["cell_line"] = summarize_adata(
            cell_line_path,
            CELL_LINE_REQUIRED_OBS,
        )
        validate_summary(report["cell_line"], "cell_line", report["errors"])
        cell_counts = report["cell_line"].get("cell_counts", {})
        if cell_counts and min(cell_counts.values()) < 50:
            report["errors"].append("cell_line: a retained line has fewer than 50 cells")
    except Exception as exc:
        report["errors"].append(f"cell_line: could not read or validate file: {exc}")

    try:
        report["tumor"] = summarize_adata(tumor_path, TUMOR_REQUIRED_OBS)
        validate_summary(report["tumor"], "tumor", report["errors"])
        cell_type_counts = report["tumor"].get("value_counts", {}).get("cell_type", {})
        if set(cell_type_counts) != {"EpiT"}:
            report["warnings"].append(
                "tumor: processed data contain cell types other than EpiT; "
                "inspect the reported value_counts"
            )
    except Exception as exc:
        report["errors"].append(f"tumor: could not read or validate file: {exc}")

    if report["cell_line"] and report["tumor"]:
        cl = ad.read_h5ad(cell_line_path, backed="r")
        tumor = ad.read_h5ad(tumor_path, backed="r")
        try:
            cl_names = set(cl.var_names)
            tumor_names = set(tumor.var_names)
            report["cross_dataset"] = {
                "same_gene_count": len(cl_names & tumor_names),
                "cell_line_only_genes": len(cl_names - tumor_names),
                "tumor_only_genes": len(tumor_names - cl_names),
                "same_gene_order": bool(list(cl.var_names) == list(tumor.var_names)),
            }
            if cl_names != tumor_names:
                report["errors"].append(
                    "cross_dataset: processed AnnData objects do not have the same gene set"
                )
            if report["genes"] and report["genes"].get("rows") != cl.n_vars:
                report["errors"].append(
                    "cross_dataset: gene table row count differs from processed AnnData gene count"
                )
        finally:
            cl.file.close()
            tumor.file.close()

    report["status"] = "error" if report["errors"] else "warning" if report["warnings"] else "ok"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print(f"Preprocessing validation report: {report_path}")
    print(f"Status: {report['status']}")
    for message in report["warnings"]:
        print(f"WARNING: {message}")
    for message in report["errors"]:
        print(f"ERROR: {message}")

    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
