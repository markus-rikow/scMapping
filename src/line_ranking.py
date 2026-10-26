"""Equal-weight cell-line aggregation for post-HPO model ranking.

The HPO summary rows average over all cell-line cells, so larger cell lines
dominate the global mean. These helpers instead summarize the per-cell kNN
mixing table into one row per cell line, then aggregate the lines with equal
weight regardless of cell count.

A cell without a tumor neighbor has undefined entropy; it is converted to
specificity 0 before averaging, so missing tumor contact cannot look specific
merely because the entropy mean ignores NaN values.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def line_metrics_from_cells(cells, mixing_col, entropy_col, coverage_col,
                            purity_col):
    """One row per cell line from a per-cell table at a single k.

    The three score components are averaged per line first; the line score is
    their geometric mean, so a low value in any component strongly penalizes
    the line. A cell without a tumor neighbor (entropy NaN) has specificity 0.
    """
    needed = ["cell_line", mixing_col, entropy_col, coverage_col, purity_col]
    missing = [c for c in needed if c not in cells.columns]
    if missing:
        raise ValueError(f"Per-cell table missing columns: {missing}")
    sub = cells[needed].copy()
    sub["specificity"] = 1.0 - sub[entropy_col].fillna(1.0)
    line = sub.groupby("cell_line", sort=False).agg(
        n_cells=("cell_line", "size"),
        mixing=(mixing_col, "mean"),
        entropy=(entropy_col, "mean"),
        specificity=("specificity", "mean"),
        coverage=(coverage_col, "mean"),
        line_purity=(purity_col, "mean"),
    ).reset_index()
    line["line_score"] = np.cbrt(
        line["mixing"] * line["specificity"] * line["line_purity"]
    )
    return line


def aggregate_line_metrics(line):
    """Equal-weight aggregation over the lines of one run at one k.

    The primary metric is the arithmetic mean of the line scores. The per-line
    means of mixing/specificity/coverage/purity are reported as diagnostics;
    the minima of line score and coverage double as worst-line diagnostics and
    ranking tie-breakers. A coverage requirement can be applied downstream.
    """
    scores = line["line_score"].astype(float)
    return {
        "mean_line_score": float(scores.mean()),
        "min_line_score": float(scores.min()),
        "mean_mixing": float(line["mixing"].astype(float).mean()),
        "mean_specificity": float(line["specificity"].astype(float).mean()),
        "mean_coverage": float(line["coverage"].astype(float).mean()),
        "min_line_coverage": float(line["coverage"].astype(float).min()),
        "mean_line_purity": float(line["line_purity"].astype(float).mean()),
    }


def rank_runs(frame, primary_col, tiebreak_cols=()):
    """Rank runs within each approach by ``primary_col`` (descending).

    Ties fall through the given ``tiebreak_cols`` (descending) and finally
    ``config_id`` (ascending). Adds a ``rank`` column, 1 = best per approach.
    """
    order = [(primary_col, False)] + [(c, False) for c in tiebreak_cols]
    order = [(c, a) for c, a in order if c in frame.columns]
    order += [("config_id", True)]
    ordered = frame.sort_values(
        [c for c, _ in order],
        ascending=[a for _, a in order],
        kind="stable",
    ).copy()
    ordered["rank"] = ordered.groupby("approach", sort=False).cumcount() + 1
    return ordered


def dynamic_k_mixing_scores(latent, obs, k_max, labels_key="label"):
    """Per-cell kNN mixing metrics with a per-cell-line k.

    Every cell-line cell uses k = min(n_cells[line], k_max), so a small line
    is never evaluated with a neighborhood larger than its own cell count and
    a large line is capped at k_max. Same metric definitions as the fixed-k
    columns of ``cell_line_mixing.csv`` (mixing, entropy, coverage,
    line_purity). Returns one row per cell-line cell.
    """
    latent = np.asarray(latent)
    tumor_mask = (obs["dataset"] == "tumor").to_numpy()
    cl_mask = ~tumor_mask
    cl_idx = np.where(cl_mask)[0]

    cl_lines = obs["cell_line"].to_numpy()[cl_idx]
    line_sizes = pd.Series(cl_lines).value_counts().to_dict()
    per_cell_k = np.array(
        [min(line_sizes[ln], k_max) for ln in cl_lines], dtype=int
    )
    max_k = int(per_cell_k.max())

    nn = NearestNeighbors(n_neighbors=max_k + 1, n_jobs=1)
    nn.fit(latent)
    _, neighbor_idx = nn.kneighbors(latent[cl_idx])
    neighbor_idx = neighbor_idx[:, 1:]  # self is the nearest neighbor

    cl_nn = NearestNeighbors(n_neighbors=max_k + 1, n_jobs=1)
    cl_nn.fit(latent[cl_idx])
    _, cl_neighbor_idx = cl_nn.kneighbors(latent[cl_idx])
    cl_neighbor_idx = cl_neighbor_idx[:, 1:]
    cl_neighbor_lines = cl_lines[cl_neighbor_idx]

    n_labels = obs.loc[tumor_mask, labels_key].nunique()
    if n_labels < 2:
        raise ValueError(
            "Need at least 2 tumor labels for entropy normalization."
        )

    neighbor_labels = obs[labels_key].to_numpy()[neighbor_idx]

    records = []
    for row, gidx in enumerate(cl_idx):
        k = int(per_cell_k[row])
        nbrs = neighbor_idx[row, :k]
        is_tumor = tumor_mask[nbrs]
        mixing = float(is_tumor.sum()) / k
        tumor_labels = neighbor_labels[row, :k][is_tumor]
        record = {
            "obs_index": obs.index[gidx],
            "cell_line": cl_lines[row],
            "batch": obs["batch"].iloc[gidx],
        }
        record["mixing"] = mixing
        record["coverage"] = int(mixing > 0)
        if len(tumor_labels) > 0:
            counts = np.bincount(
                pd.Categorical(tumor_labels).codes, minlength=n_labels
            )
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            record["entropy"] = -np.sum(probs * np.log(probs)) / np.log(
                n_labels
            )
        else:
            record["entropy"] = np.nan
        record["line_purity"] = float(
            (cl_neighbor_lines[row, :k] == cl_lines[row]).mean()
        )
        records.append(record)
    return pd.DataFrame(records)