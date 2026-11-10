"""Group-level concordance and cross-group distance/enrichment helpers."""

import re

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.bio.matching import tumor_group_sort_key


def group_profiles(scores, group_values, index):
    groups = pd.Series(group_values, index=index, name="group")
    profiles = (
        scores.join(groups).groupby("group", observed=True)[scores.columns].median()
    )
    profiles.index = profiles.index.str.replace("^line_", "", regex=True)
    return profiles


def full_concordance(profiles, order):
    """Spearman correlation across all groups in the requested order."""
    return profiles.loc[order].T.corr(method="spearman")


def zscored_concordance(profiles, order):
    """Pearson correlation of pathway-z-scored group profiles."""
    p = profiles.loc[order].astype(float)
    sd = p.std(axis=0, ddof=0)
    keep = sd > 0
    z = (p.loc[:, keep] - p.loc[:, keep].mean(axis=0)) / sd[keep]
    return z.T.corr(method="pearson")


def block_summary(matrix, line_order, tumor_order):
    """Mean and minimum within-cell-line, cross-domain, and within-tumor blocks."""
    cell_line_block = matrix.loc[line_order, line_order].to_numpy()
    cell_line_block = cell_line_block[~np.eye(len(line_order), dtype=bool)]
    tumor_block = matrix.loc[tumor_order, tumor_order].to_numpy()
    tumor_block = tumor_block[~np.eye(len(tumor_order), dtype=bool)]
    cross_block = matrix.loc[line_order, tumor_order].to_numpy().ravel()
    return pd.Series(
        {
            "cell line mean": cell_line_block.mean(),
            "cell line-tumor mean": cross_block.mean(),
            "tumor mean": tumor_block.mean(),
            "cell line min": cell_line_block.min(),
            "cell line-tumor min": cross_block.min(),
            "tumor min": tumor_block.min(),
        }
    ).round(3)


def _short_label(value):
    match = re.search(r"cE\d+", str(value))
    return match.group(0) if match else str(value)


def dominant_labels(obs, group_values, dataset_col="dataset", label_col="label"):
    """Dominant Pelka epithelial state label and share for each tumor subpopulation."""
    is_tumor = obs[dataset_col].eq("tumor").to_numpy()
    groups = np.asarray(group_values, dtype=str)
    pelka = obs[label_col].astype(str).to_numpy()
    rows = []
    for group in pd.unique(groups[is_tumor]):
        values = pelka[groups == group]
        mode = pd.Series(values).mode().iloc[0]
        rows.append(
            {
                "group": group,
                "n_cells": int(values.size),
                "dominant": _short_label(mode),
                "share": round(float((values == mode).mean()), 2),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("group", key=lambda s: s.map(tumor_group_sort_key))
        .set_index("group")
    )


def bare_group_labels(group_values):
    """Strip the 'line_' prefix from group labels."""
    return np.asarray(pd.Series(group_values).str.replace(r"^line_", "", regex=True))


def symmetric_nn_distance(latent, labels, group_list, n=80, seed=0):
    """Size-matched symmetric median nearest-neighbor Euclidean distance across groups."""
    latent = np.asarray(latent)
    labels = np.asarray(labels, dtype=str)
    rng = np.random.default_rng(seed)
    sampled = {}
    for group in group_list:
        members = np.where(labels == group)[0]
        if members.size < n:
            raise ValueError(f"Group {group!r} has only {members.size} cells (< {n})")
        sampled[group] = latent[rng.choice(members, size=n, replace=False)]

    distance = pd.DataFrame(np.nan, index=group_list, columns=group_list)
    for target in group_list:
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(sampled[target])
        for source in group_list:
            if source == target:
                continue
            d, _ = nn.kneighbors(sampled[source])
            distance.loc[source, target] = float(np.median(d[:, 0]))
        nn_self = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(
            sampled[target]
        )
        d_self, _ = nn_self.kneighbors(sampled[target])
        distance.loc[target, target] = float(np.median(d_self[:, 1]))
    return (distance + distance.T) / 2


def all_vs_all_knn_oe(latent, labels, group_list, k=50):
    """Symmetric log2(observed / expected) kNN neighborhood enrichment across groups."""
    latent = np.asarray(latent)
    labels = np.asarray(labels, dtype=str)
    sizes = {group: int((labels == group).sum()) for group in group_list}
    oe = pd.DataFrame(np.nan, index=group_list, columns=group_list)
    for source in group_list:
        source_mask = labels == source
        pool_mask = ~source_mask
        pool_groups = labels[pool_mask]
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(latent[pool_mask])
        _, idx = nn.kneighbors(latent[source_mask])
        observed = pd.Series(pool_groups[idx].ravel()).value_counts(normalize=True)
        pool_size = int(pool_mask.sum())
        for target in group_list:
            if target == source:
                continue
            observed_frac = float(observed.get(target, 0.0))
            expected_frac = sizes[target] / pool_size
            if observed_frac > 0:
                oe.loc[source, target] = np.log2(observed_frac / expected_frac)
    both = oe.to_numpy()
    count = np.isfinite(both).astype(float) + np.isfinite(both.T).astype(float)
    total = np.nan_to_num(both, nan=0.0) + np.nan_to_num(both.T, nan=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        symmetric = np.where(count > 0, total / count, np.nan)
    return pd.DataFrame(symmetric, index=group_list, columns=group_list)
