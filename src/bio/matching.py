"""Tumor-only kNN matching: composition, candidate calls, distances, and confidence flags."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

META_COLUMNS = {"cell_line", "nearest_tumor_distance"}


def tumor_group_sort_key(group):
    return int(str(group).rsplit("_", 1)[-1])


def make_match_groups(obs, tumor_group_col, dataset_col="dataset", cell_line_col="cell_line"):
    """Assign tumor cells to their subpopulation and cell line cells to their line."""
    is_tumor = obs[dataset_col].eq("tumor")
    is_line = obs[dataset_col].eq("cell_line")
    groups = pd.Series(pd.NA, index=obs.index, dtype="string", name="match_group")
    groups.loc[is_tumor] = (
        "tumor_" + obs.loc[is_tumor, tumor_group_col].astype("string")
    )
    groups.loc[is_line] = (
        "line_" + obs.loc[is_line, cell_line_col].astype("string")
    )
    if groups.isna().any():
        raise ValueError("Every cell must belong to a tumor subpopulation or a cell line")
    return groups


def knn_composition(latent, groups, obs, k, dataset_col="dataset", cell_line_col="cell_line"):
    """Composition of each cell line cell's k nearest tumor cells.

    Returns one row per cell line cell, with the mean composition columns and
    the nearest distance to each tumor subpopulation among those neighbors.
    """
    latent = np.asarray(latent)
    groups = np.asarray(groups, dtype=str)
    is_tumor = obs[dataset_col].eq("tumor").to_numpy()
    is_line = ~is_tumor
    tumor_latent = latent[is_tumor]
    line_latent = latent[is_line]
    tumor_groups = groups[is_tumor]

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(tumor_latent)
    dist, idx = nn.kneighbors(line_latent)
    neighbor_groups = tumor_groups[idx]
    group_order = sorted(pd.unique(tumor_groups), key=tumor_group_sort_key)

    composition = pd.DataFrame(
        {group: (neighbor_groups == group).mean(axis=1) for group in group_order},
        index=obs.index[is_line],
    )
    composition.insert(
        0, cell_line_col, obs.loc[is_line, cell_line_col].astype(str).to_numpy()
    )
    composition["nearest_tumor_distance"] = dist[:, 0]
    for group in group_order:
        member = neighbor_groups == group
        composition[f"dist_{group}"] = np.where(
            member.any(axis=1),
            np.where(member, dist, np.inf).min(axis=1),
            np.nan,
        )
    return composition, group_order


def tumor_tumor_baseline(latent, obs, dataset_col="dataset", quantile=0.95):
    """95th percentile of the tumor-to-tumor nearest-neighbor distance."""
    is_tumor = obs[dataset_col].eq("tumor").to_numpy()
    tumor_latent = np.asarray(latent)[is_tumor]
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(tumor_latent)
    distances = nn.kneighbors(tumor_latent)[0][:, 1]
    return float(np.quantile(distances, quantile))


def _median(series):
    values = series.dropna()
    return float(values.median()) if len(values) else np.nan


def summarize_matches(composition, baseline, margin=0.10, cell_line_col="cell_line"):
    """Collapse per-cell composition to one candidate row per cell line."""
    group_columns = [
        column
        for column in composition.columns
        if column not in META_COLUMNS
        and not column.startswith("dist_")
        and column != cell_line_col
    ]
    rows = []
    for cell_line, cells in composition.groupby(cell_line_col, observed=True, sort=False):
        mean_composition = cells[group_columns].mean().sort_values(ascending=False)
        primary, second = mean_composition.index[:2]
        margin_value = float(mean_composition.iloc[0] - mean_composition.iloc[1])
        median_nearest = _median(cells["nearest_tumor_distance"])
        rows.append(
            {
                cell_line_col: cell_line,
                "n_cells": int(cells.shape[0]),
                "primary_match": primary,
                "primary_frac": float(mean_composition.iloc[0]),
                "second_match": second,
                "second_frac": float(mean_composition.iloc[1]),
                "margin": margin_value,
                "multiple_viable": margin_value < margin,
                "median_dist_nearest_tumor": median_nearest,
                "median_dist_primary": _median(cells[f"dist_{primary}"]),
                "median_dist_second": _median(cells[f"dist_{second}"]),
                "frac_cells_matched_neighbor": float(
                    cells[f"dist_{primary}"].notna().mean()
                ),
                "low_proximity": median_nearest > baseline,
            }
        )
    return pd.DataFrame(rows).set_index(cell_line_col).sort_index()


def match_lines(
    latent,
    groups,
    obs,
    k,
    margin=0.10,
    dataset_col="dataset",
    cell_line_col="cell_line",
):
    """Convenience wrapper returning composition, matches, baseline, and group order."""
    composition, group_order = knn_composition(
        latent, groups, obs, k, dataset_col=dataset_col, cell_line_col=cell_line_col
    )
    baseline = tumor_tumor_baseline(latent, obs, dataset_col=dataset_col)
    matches = summarize_matches(
        composition, baseline, margin=margin, cell_line_col=cell_line_col
    )
    return composition, matches, baseline, group_order


def line_order(matches):
    """Stable row order: primary subpopulation first, then decreasing margin."""
    return sorted(
        matches.index,
        key=lambda cell_line: (
            tumor_group_sort_key(matches.loc[cell_line, "primary_match"]),
            -float(matches.loc[cell_line, "margin"]),
        ),
    )


def k_sensitivity(
    latent,
    groups,
    obs,
    k_values,
    margin=0.10,
    dataset_col="dataset",
    cell_line_col="cell_line",
):
    """Run matching for several neighborhood sizes on one baseline."""
    baseline = tumor_tumor_baseline(latent, obs, dataset_col=dataset_col)
    by_k = {}
    frames = []
    for k in k_values:
        composition, group_order = knn_composition(
            latent, groups, obs, k, dataset_col=dataset_col, cell_line_col=cell_line_col
        )
        matches = summarize_matches(
            composition, baseline, margin=margin, cell_line_col=cell_line_col
        )
        by_k[k] = (composition, matches)
        frames.append(matches.assign(k=k).reset_index())
    return by_k, pd.concat(frames, ignore_index=True), baseline


def k_sensitivity_tables(by_k, k_summary, reference_k=50):
    """Build the k-sensitivity summary, stability, comparison, and overview tables."""
    cols = [
        "k",
        "cell_line",
        "primary_match",
        "primary_frac",
        "second_match",
        "second_frac",
        "margin",
        "multiple_viable",
        "low_proximity",
    ]
    summary_table = k_summary[cols].sort_values(["k", "margin"]).reset_index(drop=True)

    primary_by_k = (
        summary_table.pivot(index="cell_line", columns="k", values="primary_match")
        .rename(columns=lambda k: f"primary_k{k}")
    )
    primary_by_k["n_unique_primary"] = primary_by_k.nunique(axis=1)
    primary_by_k["stable_across_all_k"] = primary_by_k["n_unique_primary"].eq(1)

    k_comparison = None
    if reference_k in by_k:
        ref_matches = by_k[reference_k][1][
            ["primary_match", "margin", "multiple_viable"]
        ].rename(
            columns={
                "primary_match": f"primary_k{reference_k}",
                "margin": f"margin_k{reference_k}",
                "multiple_viable": f"multiple_viable_k{reference_k}",
            }
        )
        comp = summary_table.loc[summary_table["k"].ne(reference_k)].merge(
            ref_matches, left_on="cell_line", right_index=True
        )
        comp[f"primary_changed_vs_k{reference_k}"] = (
            comp["primary_match"] != comp[f"primary_k{reference_k}"]
        )
        comp[f"margin_change_vs_k{reference_k}"] = (
            comp["margin"] - comp[f"margin_k{reference_k}"]
        )
        k_comparison = comp[
            [
                "k",
                "cell_line",
                "primary_match",
                f"primary_k{reference_k}",
                f"primary_changed_vs_k{reference_k}",
                "margin",
                f"margin_k{reference_k}",
                f"margin_change_vs_k{reference_k}",
                "multiple_viable",
                "low_proximity",
            ]
        ].sort_values(["k", "cell_line"]).reset_index(drop=True)

    k_overview = summary_table.groupby("k", as_index=True).agg(
        n_lines=("cell_line", "nunique"),
        n_multiple_viable=("multiple_viable", "sum"),
        n_low_proximity=("low_proximity", "sum"),
        median_margin=("margin", "median"),
        minimum_margin=("margin", "min"),
    )
    return summary_table, primary_by_k, k_comparison, k_overview


def subpopulation_distances(
    latent,
    groups,
    obs,
    tumor_order=None,
    row_order=None,
    dataset_col="dataset",
    cell_line_col="cell_line",
):
    """Median nearest-cell Euclidean distance from each cell line to each tumor subpopulation."""
    latent = np.asarray(latent)
    groups = np.asarray(groups, dtype=str)
    is_tumor = obs[dataset_col].eq("tumor").to_numpy()
    is_line = ~is_tumor
    line_latent = latent[is_line]

    if tumor_order is None:
        tumor_order = sorted(pd.unique(groups[is_tumor]), key=tumor_group_sort_key)

    cell_distances = pd.DataFrame(index=obs.index[is_line])
    for tumor_group in tumor_order:
        group_mask = is_tumor & (groups == tumor_group)
        nn_group = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(
            latent[group_mask]
        )
        cell_distances[tumor_group] = nn_group.kneighbors(
            line_latent, return_distance=True
        )[0][:, 0]

    cell_distances[cell_line_col] = (
        obs.loc[is_line, cell_line_col].astype(str).to_numpy()
    )
    matrix = cell_distances.groupby(cell_line_col, observed=True)[tumor_order].median()
    if row_order is not None:
        matrix = matrix.loc[row_order]
    return matrix

