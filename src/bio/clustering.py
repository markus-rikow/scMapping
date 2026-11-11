"""Tumor-only and full-graph Leiden resolution sweeps, seed selection, and artifact builder."""

from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

RAW_META_COLS = ["LymphNodeStatus", "TissueSource", "ProcessingMethod", "TumorStage"]
DROP_OBS = ["cell_type", "c_value", "pool_id"]


def build_combined_full_adata(
    adata_tumor,
    adata_cl,
    latent,
    obs_names,
    hvg_genes,
    tumor_metadata_path=None,
):
    """Build the full-gene combined AnnData with raw counts, log1p X, QC metrics, and HVG flag."""
    combined_full = ad.concat([adata_cl, adata_tumor])
    if list(combined_full.obs_names) != list(obs_names):
        raise ValueError("Combined obs_names do not match the latent run obs_names")

    combined_full.obs = (
        pd.concat([adata_cl.obs, adata_tumor.obs], axis=0)
        .reindex(combined_full.obs_names)
    )

    if tumor_metadata_path is not None and Path(tumor_metadata_path).exists():
        raw_meta = (
            pd.read_csv(
                tumor_metadata_path,
                sep="\t",
                usecols=["NAME"] + RAW_META_COLS,
                low_memory=False,
            )
            .iloc[1:]  # drop the stray header row in the source file
            .rename(columns={"NAME": "cell_id"})
            .set_index("cell_id")
        )
        combined_full.obs = combined_full.obs.join(raw_meta, on="cell_id")

    combined_full.obs = combined_full.obs.drop(
        columns=[c for c in DROP_OBS if c in combined_full.obs.columns]
    )

    # Bool columns (e.g. mlh1_status) gain NaN on the other dataset, which turns
    # them into plain object. h5py cannot write that, so keep them categorical.
    for col in combined_full.obs.columns:
        if combined_full.obs[col].dtype == object:
            non_null = combined_full.obs[col].dropna()
            if len(non_null) and not all(isinstance(v, str) for v in non_null.unique()):
                combined_full.obs[col] = combined_full.obs[col].astype("category")

    combined_full.obsm["X_latent"] = np.asarray(latent)
    combined_full.layers["counts"] = combined_full.X.copy()

    combined_full.var["mt"] = combined_full.var_names.str.startswith(("MT-", "mt-"))
    combined_full.var["ribo"] = combined_full.var_names.str.startswith(("RPS", "RPL"))
    sc.pp.calculate_qc_metrics(
        combined_full,
        qc_vars=["mt", "ribo"],
        percent_top=None,
        log1p=False,
        layer="counts",
        inplace=True,
    )
    sc.pp.normalize_total(combined_full, target_sum=1e4)
    sc.pp.log1p(combined_full)

    combined_full.var["hvg"] = combined_full.var_names.isin(list(hvg_genes))
    return combined_full


def run_tumor_leiden_sweep(combined_full, tumor_adata, resolutions, sweep_seed=0):
    """Run Leiden across resolutions on the tumor-only graph and copy labels to combined_full."""
    rows = []
    for res in resolutions:
        key = f"tumor_leiden_res{res}"
        sc.tl.leiden(
            tumor_adata,
            resolution=res,
            key_added=key,
            random_state=sweep_seed,
            flavor="igraph",
            n_iterations=2,
        )
        labels = pd.Series(
            tumor_adata.obs[key].astype(str).to_numpy(),
            index=tumor_adata.obs_names,
        ).reindex(combined_full.obs_names)
        combined_full.obs[key] = pd.Categorical(
            labels, categories=sorted(labels.dropna().unique(), key=int)
        )
        vc = tumor_adata.obs[key].value_counts()
        rows.append(
            {
                "resolution": res,
                "n_clusters": int(vc.size),
                "min_size": int(vc.min()),
                "max_size": int(vc.max()),
            }
        )
    return pd.DataFrame(rows)


def run_full_leiden_sweep(combined_full, resolutions, sweep_seed=0):
    """Run Leiden across resolutions on the combined cell-line + tumor graph."""
    rows = []
    for res in resolutions:
        key = f"full_leiden_res{res}"
        sc.tl.leiden(
            combined_full,
            resolution=res,
            key_added=key,
            random_state=sweep_seed,
            flavor="igraph",
            n_iterations=2,
        )
        vc = combined_full.obs[key].value_counts()
        rows.append(
            {
                "resolution": res,
                "n_clusters": int(vc.size),
                "min_size": int(vc.min()),
                "max_size": int(vc.max()),
            }
        )
    return pd.DataFrame(rows)


def score_leiden(adata, leiden_key):
    """Compute cluster size, Pelka AMI, label concentration, and line subgroup counts."""
    valid = adata.obs[leiden_key].notna()
    clusters = adata.obs.loc[valid, leiden_key].astype(str)
    cluster_sizes = clusters.value_counts()

    known_labels = (
        valid
        & adata.obs["label"].notna()
        & adata.obs["label"].ne("Unknown")
    )
    labels = adata.obs.loc[known_labels, "label"].astype(str)
    known_clusters = adata.obs.loc[known_labels, leiden_key].astype(str)

    label_table = pd.crosstab(labels, known_clusters)
    balanced_label_concentration = (
        label_table.max(axis=1) / label_table.sum(axis=1)
    ).mean()

    line_mask = (
        valid
        & adata.obs["cell_line"].notna()
        & adata.obs["cell_line"].ne("Unknown")
        & adata.obs["cell_line"].ne("tumor")
    )
    line_data = pd.DataFrame(
        {
            "line": adata.obs.loc[line_mask, "cell_line"].astype(str),
            "cluster": adata.obs.loc[line_mask, leiden_key].astype(str),
        }
    )
    line_subgroup_sizes = line_data.value_counts()

    return pd.Series(
        {
            "leiden_key": leiden_key,
            "n_clusters": int(clusters.nunique()),
            "min_cluster_size": int(cluster_sizes.min()),
            "median_cluster_size": float(cluster_sizes.median()),
            "max_cluster_size": int(cluster_sizes.max()),
            "small_clusters_lt_50": int((cluster_sizes < 50).sum()),
            "adjusted_mutual_info": float(
                adjusted_mutual_info_score(labels, known_clusters)
            ),
            "balanced_label_concentration": float(balanced_label_concentration),
            "line_subgroups_le_50": int((line_subgroup_sizes <= 50).sum()),
            "line_subgroups_gt_50": int((line_subgroup_sizes > 50).sum()),
        }
    )


def score_leiden_sweep(adata, prefixes=("tumor_leiden_", "full_leiden_")):
    """Score all Leiden columns matching the given prefixes."""
    keys = [
        key for key in adata.obs.columns if key.startswith(tuple(prefixes))
    ]
    return pd.DataFrame([score_leiden(adata, key) for key in keys])


def cluster_audit(adata, cluster_key):
    """Per-cluster composition audit across cell lines and Pelka state labels."""
    obs = adata.obs.copy()
    valid = obs[cluster_key].notna()

    data = obs.loc[valid, [cluster_key, "cell_line", "label"]].copy()
    data["cluster"] = data[cluster_key].astype(str)
    data["label"] = data["label"].fillna("Unknown").astype(str)

    cluster_sizes = data["cluster"].value_counts().rename("n_cells")
    line_counts = pd.crosstab(data["cluster"], data["cell_line"])
    label_counts = pd.crosstab(data["cluster"], data["label"])

    audit = cluster_sizes.to_frame()
    audit["n_tumor"] = line_counts.get("tumor", 0)
    audit["tumor_fraction"] = audit["n_tumor"] / audit["n_cells"]
    audit["dominant_cell_line"] = line_counts.idxmax(axis=1)
    audit["dominant_cell_line_fraction"] = line_counts.max(axis=1) / audit["n_cells"]
    audit["dominant_label"] = label_counts.idxmax(axis=1)
    audit["dominant_label_fraction"] = label_counts.max(axis=1) / audit["n_cells"]
    audit["n_cell_lines"] = (line_counts > 0).sum(axis=1)
    audit["n_labels"] = (label_counts > 0).sum(axis=1)

    return audit.sort_index(key=lambda idx: idx.map(int))


def line_subgroups_table(adata, full_key, min_cells=50):
    """Cell counts of each (cell_line, full_cluster) pair and usability flag."""
    line_mask = adata.obs["dataset"].eq("cell_line")
    subgroups = (
        pd.DataFrame(
            {
                "line": adata.obs.loc[line_mask, "cell_line"].astype(str),
                "full_cluster": adata.obs.loc[line_mask, full_key].astype(str),
            }
        )
        .value_counts()
        .rename("n_cells")
        .reset_index()
    )
    subgroups["usable"] = subgroups["n_cells"] >= min_cells
    return subgroups


def leiden_seed_partitions(adata, resolution, seeds, key="leiden_check"):
    """Run Leiden on a fixed neighbor graph across multiple random seeds."""
    partitions = {}
    for seed in seeds:
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=key,
            random_state=seed,
            flavor="igraph",
            n_iterations=2,
        )
        partitions[seed] = adata.obs[key].astype(str).to_numpy()
    return partitions


def seed_stability_tables(partitions):
    """Summarize cluster size distributions and pairwise ARI/AMI across seeds."""
    cluster_rows = []
    for seed, labels in partitions.items():
        sizes = pd.Series(labels).value_counts()
        cluster_rows.append(
            {
                "seed": seed,
                "n_clusters": int(sizes.size),
                "min_cluster_size": int(sizes.min()),
                "median_cluster_size": float(sizes.median()),
                "max_cluster_size": int(sizes.max()),
            }
        )
    cluster_summary = pd.DataFrame(cluster_rows)

    pair_rows = []
    for seed_a, seed_b in combinations(partitions, 2):
        pair_rows.append(
            {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "ARI": float(
                    adjusted_rand_score(partitions[seed_a], partitions[seed_b])
                ),
                "AMI": float(
                    adjusted_mutual_info_score(partitions[seed_a], partitions[seed_b])
                ),
            }
        )
    pairwise = pd.DataFrame(pair_rows)

    mean_ari = (
        pd.concat(
            [
                pairwise[["seed_a", "ARI"]].rename(columns={"seed_a": "seed"}),
                pairwise[["seed_b", "ARI"]].rename(columns={"seed_b": "seed"}),
            ]
        )
        .groupby("seed")["ARI"]
        .mean()
        .rename("mean_ARI")
    )

    return cluster_summary, pairwise, mean_ari


def tumor_seed_metrics(partitions, adata, mean_ari):
    """Per-seed cluster size, max share, Pelka AMI, and mixed-cluster diagnostics."""
    known = adata.obs["label"].notna() & adata.obs["label"].ne("Unknown")
    known_labels = adata.obs.loc[known, "label"].astype(str)

    rows = []
    for seed, labels in partitions.items():
        partition = pd.Series(labels, index=adata.obs_names)
        sizes = partition.value_counts()
        label_table = pd.crosstab(known_labels, partition[known])
        label_fractions = label_table / label_table.sum(axis=0)
        rows.append(
            {
                "seed": seed,
                "n_clusters": int(sizes.size),
                "min_cluster_size": int(sizes.min()),
                "max_cluster_size": int(sizes.max()),
                "max_cluster_share": float(sizes.max() / sizes.sum()),
                "ami_vs_pelka": float(
                    adjusted_mutual_info_score(known_labels, partition[known])
                ),
                "mixed_clusters_ge2_labels": int(
                    (label_fractions.ge(0.20).sum(axis=0) >= 2).sum()
                ),
                "mean_ARI": float(mean_ari.loc[seed]),
            }
        )
    return pd.DataFrame(rows).set_index("seed")


def select_leiden_seed(
    seed_selection, min_cluster_size=50, max_cluster_share=0.15
):
    """Select the least over-merged seed that passes fragment and max-share screens."""
    has_fragments = seed_selection["min_cluster_size"] < min_cluster_size
    over_merged = seed_selection["max_cluster_share"] > max_cluster_share
    valid_seeds = seed_selection.loc[~has_fragments & ~over_merged]

    if valid_seeds.empty:
        raise ValueError(
            "No seed passes the cluster-size screens. Review the resolution."
        )

    selected_seed = int(
        valid_seeds.sort_values(
            ["max_cluster_share", "ami_vs_pelka"], ascending=[True, False]
        ).index[0]
    )
    table = seed_selection.copy()
    table["selected"] = table.index == selected_seed
    rejected_fragments = list(seed_selection.loc[has_fragments].index)
    rejected_over_merged = list(seed_selection.loc[over_merged].index)
    return selected_seed, table, rejected_fragments, rejected_over_merged


def freeze_and_save_artifact(
    combined_full,
    tumor_adata,
    tumor_partitions,
    full_partitions,
    settings,
    out_path=None,
):
    """Freeze the selected tumor-only and full Leiden labels on combined_full and optionally write H5AD."""
    leiden_seed = int(settings["leiden_seed"])
    tumor_res = float(settings["tumor_res"])
    full_res = float(settings["full_res"])

    frozen_tumor = pd.Series(
        tumor_partitions[leiden_seed], index=tumor_adata.obs_names
    ).reindex(combined_full.obs_names)
    frozen_full = pd.Series(
        full_partitions[leiden_seed], index=combined_full.obs_names
    )

    t_key = f"tumor_leiden_res{tumor_res}"
    f_key = f"full_leiden_res{full_res}"
    combined_full.obs[t_key] = pd.Categorical(
        frozen_tumor, categories=sorted(frozen_tumor.dropna().unique(), key=int)
    )
    combined_full.obs[f_key] = pd.Categorical(
        frozen_full, categories=sorted(frozen_full.unique(), key=int)
    )

    combined_full.uns["bio_leiden_settings"] = dict(settings)

    frozen_keys = {t_key, f_key}
    drop_cols = [
        col
        for col in combined_full.obs.columns
        if col.startswith(
            ("tumor_leiden_", "full_leiden_", "tumor_seed_", "leiden_check")
        )
        and col not in frozen_keys
    ]
    combined_full.obs = combined_full.obs.drop(columns=drop_cols)

    uns_keep = ["log1p", "neighbors", "umap", "bio_leiden_settings"]
    combined_full.uns = {
        key: combined_full.uns[key]
        for key in uns_keep
        if key in combined_full.uns
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".tmp.h5ad")
        combined_full.write_h5ad(tmp_path)
        tmp_path.replace(out_path)

    return combined_full
