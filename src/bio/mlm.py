"""Hallmark MLM scoring, cached score artifacts, and group-level summaries."""

import hashlib
import json
from pathlib import Path

import anndata as ad
import decoupler as dc
import numpy as np
import pandas as pd

PROLIFERATION_PATHWAYS = [
    "E2F_TARGETS",
    "G2M_CHECKPOINT",
    "MITOTIC_SPINDLE",
    "MYC_TARGETS_V1",
    "MYC_TARGETS_V2",
]


def array_sha256(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def build_network(hallmark_sets, genes):
    """Long-form decoupler network restricted to genes present in the data."""
    gene_set = set(map(str, genes))
    rows = []
    for pathway, pathway_genes in hallmark_sets.items():
        for gene in sorted(set(map(str, pathway_genes)) & gene_set):
            rows.append({"source": pathway, "target": gene, "weight": 1.0})
    return pd.DataFrame(rows)


def score_cells(adata, network, tmin=15, raw=False):
    """Score every cell with decoupler.mt.mlm and return scores and adjusted p-values."""
    scored = adata.copy()
    dc.mt.mlm(data=scored, net=network.copy(), tmin=tmin, raw=raw, verbose=False)
    scores = scored.obsm["score_mlm"].copy()
    padj = scored.obsm["padj_mlm"].copy()
    return scores, padj


def save_scores(
    path,
    scores,
    padj,
    obs,
    tmin=15,
    raw=False,
    extra=None,
    manifest_name="manifest.json",
):
    """Write the score artifact with a manifest and array hashes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "decoupler.mt.mlm",
        "decoupler_version": getattr(dc, "__version__", "unknown"),
        "tmin": int(tmin),
        "raw": bool(raw),
        "n_cells": int(scores.shape[0]),
        "n_pathways": int(scores.shape[1]),
        "score_sha256": array_sha256(scores.to_numpy(dtype=np.float64)),
        "padj_sha256": array_sha256(padj.to_numpy(dtype=np.float64)),
    }
    if extra:
        manifest.update(extra)
    artifact = ad.AnnData(
        X=scores.to_numpy(dtype=np.float64),
        obs=obs.copy(),
        var=pd.DataFrame(index=pd.Index(scores.columns.astype(str))),
    )
    artifact.obsm["padj_mlm"] = padj.to_numpy(dtype=np.float64)
    artifact.uns["mlm_manifest"] = manifest
    artifact.write_h5ad(path, compression="gzip")
    (path.parent / manifest_name).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return artifact


def load_scores(path):
    """Load a score artifact as a DataFrame plus its obs table."""
    artifact = ad.read_h5ad(path)
    scores = pd.DataFrame(
        artifact.X,
        index=artifact.obs_names,
        columns=artifact.var_names.astype(str),
    )
    obs = artifact.obs.copy()
    if "dataset" in obs:
        obs["domain"] = obs["dataset"].astype(str)
    if "match_group" in obs:
        obs["group"] = obs["match_group"].astype(str)
    return scores, obs, artifact


def load_or_score_primary_mlm(
    combined, hallmark_sets, mlm_dir, tmin=15, raw=False, force=False
):
    """Load the primary Hallmark MLM score artifact or compute and save it."""
    mlm_dir = Path(mlm_dir)
    mlm_dir.mkdir(parents=True, exist_ok=True)
    mlm_path = mlm_dir / "hallmark_mlm_scores.h5ad"
    mlm_network_path = mlm_dir / "hallmark_network.csv"

    if force or not mlm_path.exists():
        all_genes = set(map(str, combined.var_names))
        network = build_network(hallmark_sets, all_genes)
        mlm_genes = sorted(set(network["target"]))
        adata_mlm = combined[:, mlm_genes].copy()
        scores, padj = score_cells(adata_mlm, network, tmin=int(tmin), raw=bool(raw))
        score_obs = combined.obs.loc[scores.index, ["dataset", "cell_line"]].copy()
        score_obs["match_group"] = combined.obs.loc[
            scores.index, "match_group"
        ].astype("string")
        save_scores(
            mlm_path,
            scores,
            padj,
            score_obs,
            tmin=int(tmin),
            raw=bool(raw),
            extra={
                "n_genes": len(mlm_genes),
                "genes": mlm_genes,
                "pathways": list(map(str, scores.columns)),
            },
        )
        network.to_csv(mlm_network_path, index=False)
        computed = True
    else:
        computed = False

    pathway_scores, score_obs, mlm_artifact = load_scores(mlm_path)
    return pathway_scores, score_obs, mlm_artifact, computed


def group_medians(scores, obs, domain, group_col="group", strip_line_prefix=False):
    table = scores.join(obs[["domain", group_col]])
    medians = (
        table.loc[table["domain"].eq(domain)]
        .groupby(group_col, observed=True)[scores.columns]
        .median()
    )
    if strip_line_prefix:
        medians.index = medians.index.str.replace("^line_", "", regex=True)
    return medians


def concordance_matrix(scores, obs, line_order, tumor_order, pathways=None):
    """Spearman correlation of median pathway profiles for every line/subpopulation."""
    if pathways is None:
        pathways = list(scores.columns)
    scores = scores[pathways]
    line = group_medians(scores, obs, "cell_line", strip_line_prefix=True).reindex(
        line_order
    )
    tumor = group_medians(scores, obs, "tumor").reindex(tumor_order)
    matrix = pd.DataFrame(index=line_order, columns=tumor_order, dtype=float)
    for cell_line in line_order:
        for subpopulation in tumor_order:
            matrix.loc[cell_line, subpopulation] = line.loc[cell_line].corr(
                tumor.loc[subpopulation], method="spearman"
            )
    return matrix


def domain_shift(scores, obs):
    table = scores.join(obs[["domain"]])
    medians = table.groupby("domain", observed=True)[scores.columns].median().T
    medians["difference"] = medians["cell_line"] - medians["tumor"]
    return medians.sort_values("difference", ascending=False)


def primary_runner_up(matrix, matches):
    rows = []
    for cell_line in matrix.index:
        match_key = cell_line if cell_line in matches.index else f"line_{cell_line}"
        primary = str(matches.loc[match_key, "primary_match"])
        runner_up = str(matches.loc[match_key, "second_match"])
        primary_rho = float(matrix.loc[cell_line, primary])
        runner_up_rho = float(matrix.loc[cell_line, runner_up])
        rows.append(
            {
                "cell_line": cell_line,
                "primary_match": primary,
                "runner_up": runner_up,
                "primary_rho": primary_rho,
                "runner_up_rho": runner_up_rho,
                "primary_minus_runner_up": primary_rho - runner_up_rho,
            }
        )
    return pd.DataFrame(rows).sort_values("primary_minus_runner_up").reset_index(drop=True)


def tumor_mlm_contrasts(pathway_scores, score_obs, tumor_order):
    """Tumor subpopulation median MLM score minus remaining tumor median per pathway."""
    tumor_obs = score_obs["domain"].eq("tumor")
    rows = []
    for subpopulation in tumor_order:
        source = tumor_obs & score_obs["group"].eq(subpopulation)
        rest = tumor_obs & ~score_obs["group"].eq(subpopulation)
        source_med = pathway_scores.loc[source].median()
        rest_med = pathway_scores.loc[rest].median()
        delta = source_med - rest_med
        for pathway in pathway_scores.columns:
            rows.append(
                {
                    "subpopulation": subpopulation,
                    "pathway": pathway,
                    "source_median_mlm": float(source_med[pathway]),
                    "rest_median_mlm": float(rest_med[pathway]),
                    "mlm_delta": float(delta[pathway]),
                }
            )
    return pd.DataFrame(rows)


def mlm_gsea_consistency(pathway_scores, score_obs, results_gsea_tumor, tumor_order):
    """Compare tumor subpopulation MLM deltas with GSEA NES values."""
    contrasts = tumor_mlm_contrasts(pathway_scores, score_obs, tumor_order)
    comparison = contrasts.merge(
        results_gsea_tumor[["subpopulation", "pathway", "nes", "fdr_qval"]],
        on=["subpopulation", "pathway"],
        how="inner",
    )
    comparison["same_direction"] = np.sign(comparison["mlm_delta"]) == np.sign(
        comparison["nes"]
    )
    directional = float(comparison["same_direction"].mean())
    rho = float(
        comparison[["mlm_delta", "nes"]].corr(method="spearman").loc["mlm_delta", "nes"]
    )
    return comparison, directional, rho


def pathway_similarity_matrix(line_matrix, tumor_matrix, columns, metric="spearman"):
    """Pairwise pathway-profile similarity between cell lines and tumor subpopulations."""
    lin = line_matrix.loc[:, columns]
    tum = tumor_matrix.loc[:, columns]
    keep = lin.notna().all(axis=0) & tum.notna().all(axis=0)
    lin = lin.loc[:, keep]
    tum = tum.loc[:, keep]

    if metric == "z_dist":
        diff = lin.to_numpy(float)[:, None, :] - tum.to_numpy(float)[None, :, :]
        values = -np.sqrt((diff ** 2).sum(axis=2))
    elif metric == "spearman":
        a = lin.rank(axis=1, method="average").to_numpy(float)
        b = tum.rank(axis=1, method="average").to_numpy(float)
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        values = a @ b.T
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")

    return pd.DataFrame(values, index=lin.index, columns=tum.index)


def pathway_similarity_evaluation(
    pathway_scores,
    score_obs,
    results_gsea_tumor,
    matches,
    line_order,
    tumor_order,
    prolif_arm=("tumor_4", "tumor_5", "tumor_12"),
    n_perm=1000,
    seed=0,
):
    """Evaluate raw, tumor-centered Spearman, and tumor z-space pathway similarity."""
    line_labels = [str(line).replace("line_", "") for line in line_order]
    tumor_group_scores = group_medians(pathway_scores, score_obs, "tumor").reindex(
        tumor_order
    )
    cell_line_group_scores = group_medians(
        pathway_scores, score_obs, "cell_line", strip_line_prefix=True
    ).reindex(line_labels)

    tumor_cell_mask = score_obs["domain"].eq("tumor")
    tumor_ref = pathway_scores.loc[tumor_cell_mask]
    ref_mean = tumor_ref.mean(axis=0)
    ref_median = tumor_ref.median(axis=0)
    ref_sd = tumor_ref.std(axis=0, ddof=1).replace(0.0, np.nan)

    tumor_group_z = (tumor_group_scores - ref_mean) / ref_sd
    cell_line_group_z = (cell_line_group_scores - ref_mean) / ref_sd
    tumor_group_c = tumor_group_scores - ref_median
    cell_line_group_c = cell_line_group_scores - ref_median

    sig_gsea = results_gsea_tumor[
        results_gsea_tumor["fdr_qval"].le(0.05)
        & results_gsea_tumor["nes"].abs().ge(1.0)
    ]
    all_pathways = list(tumor_group_scores.columns)
    tumor_sig_pathways = [
        p for p in all_pathways if p in set(sig_gsea["pathway"])
    ]
    prolif_arm = list(prolif_arm)
    prolif_shared = (
        sig_gsea[sig_gsea["subpopulation"].isin(prolif_arm)]
        .groupby("pathway")["nes"]
        .apply(lambda v: len(v) == len(prolif_arm) and bool((v > 0).all()))
    )
    prolif_shared_pathways = [
        p for p in tumor_sig_pathways if bool(prolif_shared.get(p, False))
    ]
    tumor_sig_noprolif = [
        p for p in tumor_sig_pathways if p not in set(prolif_shared_pathways)
    ]
    pathway_sets = {
        "all50": all_pathways,
        "tumor_sig": tumor_sig_pathways,
        "tumor_sig_noprolif": tumor_sig_noprolif,
    }

    def rank_readout(sim, label):
        rows = []
        for cell_line in line_labels:
            match_key = (
                cell_line if cell_line in matches.index else f"line_{cell_line}"
            )
            order = sim.loc[cell_line].sort_values(ascending=False).dropna()
            primary = str(matches.loc[match_key, "primary_match"])
            runner_up = str(matches.loc[match_key, "second_match"])
            primary_rank = (
                int(order.index.get_loc(primary)) + 1
                if primary in order.index
                else np.nan
            )
            runner_up_rank = (
                int(order.index.get_loc(runner_up)) + 1
                if runner_up in order.index
                else np.nan
            )
            rows.append(
                {
                    "cell_line": cell_line,
                    "method": label,
                    "primary_match": primary,
                    "primary_rank": primary_rank,
                    "runner_up": runner_up,
                    "runner_up_rank": runner_up_rank,
                    "primary_top1": primary_rank == 1,
                    "primary_top3": primary_rank <= 3,
                    "n_subpops": len(order),
                }
            )
        return pd.DataFrame(rows)

    similarity_by_method = {}
    baseline_label = "uncentered_spearman | all50 (baseline)"
    similarity_by_method[baseline_label] = pathway_similarity_matrix(
        cell_line_group_scores, tumor_group_scores, all_pathways, "spearman"
    )
    readout_frames = [rank_readout(similarity_by_method[baseline_label], baseline_label)]

    for set_name, columns in pathway_sets.items():
        if not columns:
            continue
        z_label = f"z_dist | {set_name}"
        c_label = f"centered_spearman | {set_name}"
        similarity_by_method[z_label] = pathway_similarity_matrix(
            cell_line_group_z, tumor_group_z, columns, "z_dist"
        )
        similarity_by_method[c_label] = pathway_similarity_matrix(
            cell_line_group_c, tumor_group_c, columns, "spearman"
        )
        readout_frames.append(rank_readout(similarity_by_method[z_label], z_label))
        readout_frames.append(rank_readout(similarity_by_method[c_label], c_label))

    rank_readouts = pd.concat(readout_frames, ignore_index=True)
    rank_summary = (
        rank_readouts.groupby("method", observed=True)
        .agg(
            n_lines=("cell_line", "nunique"),
            median_primary_rank=("primary_rank", "median"),
            primary_top1_rate=("primary_top1", "mean"),
            primary_top3_rate=("primary_top3", "mean"),
            median_runner_up_rank=("runner_up_rank", "median"),
        )
        .sort_values("median_primary_rank")
    )

    rng = np.random.default_rng(seed)
    null_frames = []
    for method, sim in similarity_by_method.items():
        values = sim.reindex(index=line_labels).to_numpy(float)
        primary_col = np.array(
            [
                sim.columns.get_loc(
                    str(
                        matches.loc[
                            l if l in matches.index else f"line_{l}", "primary_match"
                        ]
                    )
                )
                for l in line_labels
            ]
        )
        order = np.argsort(-values, axis=1)
        rank_of = np.empty_like(order)
        rank_of[np.arange(values.shape[0])[:, None], order] = np.arange(
            values.shape[1]
        )[None, :]
        observed = rank_of[np.arange(values.shape[0]), primary_col] + 1
        null = rng.integers(1, values.shape[1] + 1, size=(n_perm, len(line_labels)))
        null_medians = np.median(null, axis=1)
        null_frames.append(
            {
                "method": method,
                "observed_median_rank": float(np.median(observed)),
                "null_median_rank_mean": float(null_medians.mean()),
                "null_median_rank_q05": float(np.quantile(null_medians, 0.05)),
                "observed_top1_rate": float((observed == 1).mean()),
                "null_top1_rate": 1.0 / values.shape[1],
                "observed_top3_rate": float((observed <= 3).mean()),
                "null_top3_rate": float((null <= 3).mean()),
                "p_median_rank": float((null_medians <= np.median(observed)).mean()),
            }
        )
    null_summary = pd.DataFrame(null_frames).sort_values("observed_median_rank")
    return rank_readouts, rank_summary, null_summary, similarity_by_method, pathway_sets


def hvg_network_summary(hallmark_sets, all_genes, hvg_genes, tmin=15):
    """Build the HVG Hallmark network and report retained and dropped pathways at tmin."""
    all_genes = set(map(str, all_genes))
    hvg_genes = sorted(set(map(str, hvg_genes)))
    hallmark_all_genes = sorted(
        {gene for genes in hallmark_sets.values() for gene in genes} & all_genes
    )
    hvg_hallmark_genes = sorted(set(hvg_genes) & set(hallmark_all_genes))
    network_hvg = build_network(hallmark_sets, hvg_hallmark_genes)
    net_sizes = (
        network_hvg.groupby("source")["target"].nunique().sort_values(ascending=False)
    )
    hvg_pathways = sorted(net_sizes[net_sizes >= tmin].index)
    dropped_pathways = sorted(net_sizes[net_sizes < tmin].index)
    dropped_sizes = net_sizes.loc[dropped_pathways].rename("genes_in_hvg_set")
    return {
        "hallmark_all_genes": hallmark_all_genes,
        "hvg_hallmark_genes": hvg_hallmark_genes,
        "network_hvg": network_hvg,
        "hvg_pathways": hvg_pathways,
        "dropped_pathways": dropped_pathways,
        "dropped_sizes": dropped_sizes,
    }


def load_or_build_corrected_expression(
    combined,
    hvg_mask,
    hvg_hallmark_genes,
    run_dir,
    approach,
    corrected_path,
    library_size=10000,
    seed=0,
    force=False,
):
    """Load or compute scANVI-corrected log1p expression on the HVG Hallmark gene subset."""
    corrected_path = Path(corrected_path)
    manifest_path = corrected_path.with_name(f"{corrected_path.stem}_manifest.json")

    if not force and corrected_path.exists():
        return ad.read_h5ad(corrected_path), False

    import scvi

    combined_hvg = combined[:, hvg_mask].copy()
    model_dir = Path(run_dir) / ("joint_scanvi" if approach == "joint" else "query")
    model = scvi.model.SCANVI.load(str(model_dir), adata=combined_hvg)
    if not combined_hvg.var_names.equals(model.adata.var_names):
        raise ValueError("Combined HVG var_names do not match model var_names")

    scvi.settings.seed = seed
    np.random.seed(seed)
    normalized = model.get_normalized_expression(
        adata=combined_hvg,
        library_size=int(library_size),
        return_numpy=True,
    )
    gene_index = combined_hvg.var_names.get_indexer(hvg_hallmark_genes)
    if not (gene_index >= 0).all():
        raise ValueError("Missing HVG Hallmark genes in combined_hvg")
    expression = np.log1p(normalized[:, gene_index]).astype(np.float32)

    corrected = ad.AnnData(
        X=expression,
        obs=combined.obs.loc[
            combined_hvg.obs_names, ["dataset", "cell_line", "match_group"]
        ].copy(),
        var=pd.DataFrame(index=pd.Index(hvg_hallmark_genes)),
    )
    manifest = {
        "source": "scvi.model.SCANVI.get_normalized_expression",
        "model_dir": str(model_dir),
        "library_size": int(library_size),
        "transform_batch": None,
        "log1p": True,
        "n_cells": int(expression.shape[0]),
        "n_genes": int(expression.shape[1]),
        "genes": list(hvg_hallmark_genes),
        "expression_sha256": array_sha256(expression),
    }
    corrected.uns["mlm_sensitivity_manifest"] = manifest
    corrected_path.parent.mkdir(parents=True, exist_ok=True)
    corrected.write_h5ad(corrected_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return corrected, True


def concordance_metrics_by_representation(
    concordance_by_rep, line_matches, line_order, rep_labels=None
):
    """Per-representation, per-cell-line concordance metrics and aggregated summary."""
    if rep_labels is None:
        rep_labels = list(concordance_by_rep.keys())
    line_names = [str(line).replace("line_", "") for line in line_order]

    metrics_rows = []
    for label in rep_labels:
        matrix = concordance_by_rep[label]
        for cell_line in line_names:
            match_key = (
                cell_line if cell_line in line_matches.index else f"line_{cell_line}"
            )
            primary = str(line_matches.loc[match_key, "primary_match"])
            runner_up = str(line_matches.loc[match_key, "second_match"])
            order = matrix.loc[cell_line].sort_values(ascending=False)
            primary_rank = (
                int(order.index.get_loc(primary)) + 1
                if primary in order.index
                else np.nan
            )
            metrics_rows.append(
                {
                    "representation": label,
                    "cell_line": cell_line,
                    "primary_match": primary,
                    "runner_up": runner_up,
                    "primary_rho": float(matrix.loc[cell_line, primary]),
                    "runner_up_rho": float(matrix.loc[cell_line, runner_up]),
                    "primary_minus_runner_up": float(
                        matrix.loc[cell_line, primary]
                        - matrix.loc[cell_line, runner_up]
                    ),
                    "primary_rank": primary_rank,
                    "primary_top1": primary_rank == 1,
                }
            )
    metrics = pd.DataFrame(metrics_rows)
    summary = (
        metrics.groupby("representation", observed=True)
        .agg(
            n_lines=("cell_line", "nunique"),
            n_primary_gt_runner_up=(
                "primary_minus_runner_up",
                lambda v: int((v > 0).sum()),
            ),
            median_primary_rho=("primary_rho", "median"),
            min_primary_rho=("primary_rho", "min"),
            median_runner_up_rho=("runner_up_rho", "median"),
            median_primary_minus_runner_up=("primary_minus_runner_up", "median"),
            n_primary_top1=("primary_top1", "sum"),
            median_primary_rank=("primary_rank", "median"),
        )
        .reindex(rep_labels)
    )
    return metrics, summary


def cross_representation_correlations(matrices_by_rep):
    """Pairwise Spearman correlation between flattened concordance matrices."""
    flattened = pd.DataFrame(
        {
            label: matrix.to_numpy(dtype=float).ravel()
            for label, matrix in matrices_by_rep.items()
        }
    )
    return flattened.corr(method="spearman")
