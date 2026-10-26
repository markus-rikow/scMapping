"""Rank completed HPO runs by equal-weight cell-line matching scores.

Scoring: the three per-cell kNN metrics are averaged within each cell line.
The line score is the geometric mean of those three line means,

    score_line = (mixing_line * specificity_line * purity_line) ** (1/3)

where
    mixing      = tumor-neighbor fraction       (integration),
    specificity = 1 - entropy                   (specific integration),
    purity      = within-line identity fraction (consistent identity).

A cell without a tumor neighbor has undefined entropy and therefore
specificity 0; an unmixed line scores 0. The run-level score is the mean of
the ten equal-weight line scores, and the primary ranking sorts by that mean.

Two k views are computed per run:
- fixed: for each fixed neighborhood size (default 20/50/100/200), the stored
  per-cell kNN metrics are aggregated per cell line (cell_line_mixing.csv).
- dynamic: the saved model is applied to the reconstructed AnnData, its
  latent representation is extracted, and the kNN metrics are recomputed
  with k = min(n_cells(line), k_max) per cell line. Slow (loads every model);
  opt in with --dynamic.

The primary ranking uses the mean line score, at --primary-k for the fixed
view or at the dynamic k when --dynamic is passed. Joint and scArches runs are
ranked separately, and the best model is selected for each approach. scIB
Total, Batch correction, and Bio conservation scores are recorded alongside
the ranking metrics for overview and comparison, but they do not determine the
primary ranking. The saved validation ELBO is not used for model selection.

Outputs (out dir, default models/hpo):
    ranking_primary_<k>.csv  one row per run, all metrics and the per-approach
                             rank; named for the primary view, e.g.
                             ranking_primary_100.csv (fixed k=100) or
                             ranking_primary_dynamic.csv (--dynamic)
    line_metrics.csv          one row per cell line and k, per run
    best_models_primary_<k>.csv  rank-1 rows of the ranking file per
                                 approach (best_models_primary_dynamic.csv
                                 with --dynamic)

Examples:
    python scripts/05_rank_models.py              # fixed k only, primary k=100
    python scripts/05_rank_models.py --dynamic    # adds the dynamic-k view
    python scripts/05_rank_models.py --primary-k 100
"""

import argparse
import sys
from pathlib import Path

# Make the `src` package importable when run as
# `python scripts/05_rank_models.py` without PYTHONPATH being set.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import line_ranking as lr
from src import model_io
from src.utils import load_config

# results.csv columns carried into ranking.csv for overview and comparison.
METRIC_COLS = [
    "n_layers", "n_hidden", "n_latent", "dispersion", "encode_covariates",
    "n_samples_per_label", "n_epochs",
    "Total", "Batch correction", "Bio conservation",
]


def repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path):
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_dir_of(runs_dir, row):
    return (
        Path(runs_dir)
        / f"hvg{int(row['num_hvg'])}"
        / f"config_{int(row['config_id']):02d}"
        / str(row["approach"])
    )


def fixed_k_line_metrics(run_dir, k_values):
    """Per-line metrics at every fixed k from the stored per-cell table."""
    mixing = pd.read_csv(run_dir / "cell_line_mixing.csv")
    parts = []
    for k in k_values:
        line = lr.line_metrics_from_cells(
            mixing,
            f"mixing_k{k}", f"entropy_k{k}", f"coverage_k{k}",
            f"line_purity_k{k}",
        )
        line["k_label"] = f"k{k}"
        line["k_used"] = k
        parts.append(line)
    return pd.concat(parts, ignore_index=True)


def dynamic_k_line_metrics(run_dir, approach, combined, k_max, obs):
    """Per-line metrics at the dynamic k from a freshly computed latent."""
    run = model_io.load_latent_from_combined(run_dir, combined, approach)
    latent = np.asarray(run["latent"])
    if obs is None:
        obs = pd.DataFrame(
            {
                "dataset": run["dataset"],
                "cell_line": run["cell_line"],
                "label": run["label"],
                "batch": run["batch"],
            },
            index=run["obs_names"],
        )
    cells = lr.dynamic_k_mixing_scores(latent, obs, k_max)
    line = lr.line_metrics_from_cells(
        cells, "mixing", "entropy", "coverage", "line_purity"
    )
    line["k_label"] = "dyn"
    line["k_used"] = np.minimum(line["n_cells"].astype(int), k_max)
    return line


def wide_metrics(line_metrics, k_values, with_dynamic):
    """Per-run wide metrics, one column set per k variant."""
    labels = [f"k{k}" for k in k_values] + (["dyn"] if with_dynamic else [])
    wide = {}
    for label in labels:
        line = line_metrics[line_metrics["k_label"] == label]
        for key, value in lr.aggregate_line_metrics(line).items():
            wide[f"{label}_{key}"] = value
    return wide


def build_combined_obs(adata_tumor, adata_cl, num_hvg, runs_dir):
    """Combined adata + obs for one HVG level, cached across runs."""
    hvg_dir = runs_dir / f"hvg{num_hvg}"
    combined = model_io.build_combined_adata_for_hvg(
        adata_tumor, adata_cl, num_hvg, hvg_dir=hvg_dir
    )
    obs = pd.DataFrame(
        {
            "dataset": combined.obs["dataset"].to_numpy(),
            "cell_line": combined.obs["cell_line"].to_numpy(),
            "label": combined.obs["label"].to_numpy(),
            "batch": combined.obs["batch"].to_numpy(),
        },
        index=combined.obs_names,
    )
    return combined, obs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-k",
        type=int,
        default=100,
        help="Fixed k driving the primary ranking (default 100).",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Also compute the dynamic-k view by loading every saved model (slow).",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=200,
        help="Cap for the dynamic-k per-line neighborhood (default 200).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    out_dir = repo_path(cfg["paths"]["models"]) / "hpo"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(repo_path(cfg["paths"]["hpo"]["results"]))
    if results.empty:
        raise SystemExit("No runs to rank.")

    k_values = list(cfg["hpo"]["fixed"]["k_values"])
    runs_dir = repo_path(cfg["paths"]["hpo"]["runs"])
    print(f"Runs: {len(results)}; fixed k={k_values}; primary_k={args.primary_k}")

    adata_tumor = adata_cl = None
    if args.dynamic:
        import anndata as ad
        adata_tumor = ad.read_h5ad(repo_path(cfg["paths"]["processed"]["pp_tumor"]))
        adata_cl = ad.read_h5ad(repo_path(cfg["paths"]["processed"]["pp_cell_line"]))

    wide_parts = []
    line_parts = []
    hvg_cache = {}
    for _, row in results.iterrows():
        run_dir = run_dir_of(runs_dir, row)
        if not (run_dir / "DONE").exists():
            print(f"[skip] not DONE: {display_path(run_dir)}")
            continue
        line_metrics = fixed_k_line_metrics(run_dir, k_values)
        if args.dynamic:
            num_hvg = int(row["num_hvg"])
            if num_hvg not in hvg_cache:
                hvg_cache[num_hvg] = build_combined_obs(
                    adata_tumor, adata_cl, num_hvg, runs_dir
                )
            combined, obs = hvg_cache[num_hvg]
            dyn = dynamic_k_line_metrics(
                run_dir, row["approach"], combined, args.k_max, obs
            )
            line_metrics = pd.concat([line_metrics, dyn], ignore_index=True)

        meta = {
            "config_id": int(row["config_id"]),
            "num_hvg": int(row["num_hvg"]),
            "approach": row["approach"],
            "run_dir": display_path(run_dir),
        }
        for col in METRIC_COLS:
            if col in row.index:
                meta[col] = row[col]
        meta.update(wide_metrics(line_metrics, k_values, args.dynamic))

        line_metrics["config_id"] = meta["config_id"]
        line_metrics["num_hvg"] = meta["num_hvg"]
        line_metrics["approach"] = meta["approach"]
        line_metrics["run_dir"] = meta["run_dir"]
        wide_parts.append(meta)
        line_parts.append(line_metrics)
        print(f"[ok] {row['approach']} id={int(row['config_id'])} "
              f"hvg{int(row['num_hvg'])}")

    if not wide_parts:
        raise SystemExit("No runs processed.")

    ranking = pd.DataFrame(wide_parts)
    variant = "dyn" if args.dynamic else f"k{args.primary_k}"
    primary_col = f"{variant}_mean_line_score"
    if primary_col not in ranking.columns:
        raise SystemExit(
            f"Primary score column missing: {primary_col} "
            f"(use --primary-k from {k_values})."
        )
    ranking = lr.rank_runs(
        ranking,
        primary_col,
        tiebreak_cols=[f"{variant}_min_line_score", f"{variant}_min_line_coverage"],
    )
    ranking = ranking.sort_values(["approach", "rank"]).reset_index(drop=True)

    best = ranking[ranking["rank"] == 1].copy()
    ranking_name = (
        "ranking_primary_dynamic" if args.dynamic
        else f"ranking_primary_{args.primary_k}"
    )
    best_name = ranking_name.replace("ranking", "best_models")
    ranking.to_csv(out_dir / f"{ranking_name}.csv", index=False)
    pd.concat(line_parts, ignore_index=True).to_csv(
        out_dir / "line_metrics.csv", index=False
    )
    best.to_csv(out_dir / f"{best_name}.csv", index=False)

    print(f"\nPrimary: {primary_col} (per approach). Best models:")
    print(
        best[["approach", "rank", "config_id", "num_hvg", "run_dir", primary_col,
              "Total", "Batch correction", "Bio conservation"]].to_string(
            index=False
        )
    )
    print(f"\nWritten to {display_path(out_dir)}")


if __name__ == "__main__":
    main()