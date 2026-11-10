"""Differential-expression rankings and Hallmark GSEA for the plausibility analyses."""

import json
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd

GSEA_COLUMNS = {
    "Term": "pathway",
    "ES": "es",
    "NES": "nes",
    "NOM p-val": "nom_pval",
    "FDR q-val": "fdr_qval",
    "FWER p-val": "fwer_pval",
    "Tag %": "tag_pct",
    "Gene %": "gene_pct",
    "Lead_genes": "lead_genes",
}
RESULT_COLUMNS = [
    "pathway",
    "es",
    "nes",
    "nom_pval",
    "fdr_qval",
    "fwer_pval",
    "tag_pct",
    "gene_pct",
    "overlap",
    "set_size",
    "lead_genes",
]


def read_gmt(path):
    """Read a GMT file and drop the HALLMARK_ prefix from term names."""
    gene_sets = {}
    with open(path) as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            term = fields[0].replace("HALLMARK_", "")
            gene_sets[term] = [gene for gene in fields[2:] if gene]
    return gene_sets


def make_lfc_ranking(de):
    """Signed gene ranking used by preranked GSEA."""
    return (
        de[["gene", "lfc_mean"]]
        .dropna(subset=["lfc_mean"])
        .drop_duplicates("gene")
        .sort_values(["lfc_mean", "gene"], ascending=[False, True])
        .reset_index(drop=True)
    )


def run_hallmark_prerank(
    ranking,
    comparison,
    gmt_path,
    hallmark_sets,
    seed=0,
    permutations=1000,
    min_size=15,
    max_size=500,
):
    """Run gseapy.prerank and normalise its result table."""
    result = gp.prerank(
        rnk=ranking,
        gene_sets=str(gmt_path),
        seed=seed,
        method="permutation",
        permutation_num=permutations,
        min_size=min_size,
        max_size=max_size,
        weight=1,
        ascending=False,
        outdir=None,
    ).res2d.copy()
    result = result.rename(columns=GSEA_COLUMNS)
    result["pathway"] = result["pathway"].str.replace("^HALLMARK_", "", regex=True)
    result["set_size"] = result["pathway"].map(
        lambda term: len(hallmark_sets.get(term, []))
    )
    ranked_genes = set(ranking["gene"])
    result["overlap"] = result["pathway"].map(
        lambda term: len(ranked_genes & set(hallmark_sets.get(term, [])))
    )
    return result[RESULT_COLUMNS].assign(comparison=comparison)


def subpopulation_de(
    model,
    adata,
    groups,
    subpopulation,
    batch_correction=True,
    mode="change",
    delta=0.25,
):
    """DE of one tumor subpopulation against the remaining tumor cells."""
    groups = np.asarray(groups, dtype=str)
    is_tumor = adata.obs["dataset"].eq("tumor").to_numpy()
    idx1 = is_tumor & (groups == subpopulation)
    idx2 = is_tumor & (groups != subpopulation)
    de = _de(model, adata, idx1, idx2, batch_correction, mode, delta, subpopulation)
    de = de.rename(columns={"n_group1": "n_subpop", "n_group2": "n_rest"})
    return de


def contrast_de(
    model,
    adata,
    groups,
    group1,
    group2,
    batch_correction=True,
    mode="change",
    delta=0.25,
):
    """DE between two tumor subpopulations."""
    groups = np.asarray(groups, dtype=str)
    is_tumor = adata.obs["dataset"].eq("tumor").to_numpy()
    idx1 = is_tumor & (groups == group1)
    idx2 = is_tumor & (groups == group2)
    de = _de(model, adata, idx1, idx2, batch_correction, mode, delta, group1)
    de.insert(0, "group1", group1)
    de.insert(1, "group2", group2)
    return de


def _de(model, adata, idx1, idx2, batch_correction, mode, delta, label):
    de = model.differential_expression(
        adata=adata,
        batch_correction=batch_correction,
        idx1=idx1,
        idx2=idx2,
        mode=mode,
        delta=delta,
        silent=True,
    )
    de.index.name = "gene"
    de = de.reset_index()
    de.insert(0, "comparison", label)
    de.insert(1, "n_group1", int(idx1.sum()))
    de.insert(2, "n_group2", int(idx2.sum()))
    return de


def line_vs_subpopulation_de(
    model,
    adata,
    groups,
    cell_line,
    subpopulation,
    batch_correction=True,
    mode="change",
    delta=0.25,
):
    """DE of one cell line against one tumor subpopulation."""
    groups = np.asarray(groups, dtype=str)
    is_tumor = adata.obs["dataset"].eq("tumor").to_numpy()
    is_line = ~is_tumor
    idx1 = is_line & (groups == cell_line)
    idx2 = is_tumor & (groups == subpopulation)
    de = model.differential_expression(
        adata=adata,
        batch_correction=batch_correction,
        idx1=idx1,
        idx2=idx2,
        mode=mode,
        delta=delta,
        silent=True,
    )
    de.index.name = "gene"
    de = de.reset_index()
    de.insert(0, "cell_line", str(cell_line).replace("line_", ""))
    de.insert(1, "subpop", subpopulation)
    de.insert(2, "n_line", int(idx1.sum()))
    de.insert(3, "n_subpop", int(idx2.sum()))
    return de


def de_qc(de_by_group, group_name="subpopulation"):
    """One quality-control row per DE table."""
    rows = []
    for group, de in de_by_group.items():
        if "n_line" in de.columns:
            n_source = int(de["n_line"].iloc[0])
            n_rest = int(de["n_subpop"].iloc[0])
        elif "n_subpop" in de.columns:
            n_source = int(de["n_subpop"].iloc[0])
            n_rest = int(de["n_rest"].iloc[0]) if "n_rest" in de.columns else int(de["n_subpop"].iloc[0])
        else:
            n_source = int(de["n_group1"].iloc[0])
            n_rest = int(de["n_group2"].iloc[0])
        rows.append(
            {
                group_name: group,
                "n_source": n_source,
                "n_rest": n_rest,
                "n_genes": int(de["lfc_mean"].notna().sum()),
                "n_de_fdr_0.05": int(de["is_de_fdr_0.05"].sum()),
                "n_up_fdr_0.05": int(
                    (de["is_de_fdr_0.05"] & (de["lfc_mean"] > 0)).sum()
                ),
                "n_down_fdr_0.05": int(
                    (de["is_de_fdr_0.05"] & (de["lfc_mean"] < 0)).sum()
                ),
                "max_proba_de": float(de["proba_de"].max()),
                "median_abs_lfc": float(de["lfc_mean"].abs().median()),
                "top_up": ", ".join(de.nlargest(10, "lfc_mean")["gene"]),
                "top_down": ", ".join(de.nsmallest(10, "lfc_mean")["gene"]),
            }
        )
    return pd.DataFrame(rows)


def _ensure_model_and_adata(model, adata):
    if "hvg" in adata.var.columns and not adata.var["hvg"].astype(bool).all():
        hvg_mask = adata.var["hvg"].astype(bool).to_numpy()
        adata_view = adata[:, hvg_mask]
        adata_hvg = adata_view.to_memory() if getattr(adata, "isbacked", False) else adata_view.copy()
    else:
        adata_hvg = adata.to_memory() if getattr(adata, "isbacked", False) else adata
    if isinstance(model, (str, Path)):
        import scvi

        loaded = scvi.model.SCANVI.load(str(model), adata=adata_hvg)
        if not adata_hvg.var_names.equals(loaded.adata.var_names):
            raise ValueError("HVG var_names do not match model var_names")
        return loaded, adata_hvg
    return model, adata_hvg


def load_or_run_tumor_de(
    model,
    adata,
    groups_de,
    tumor_subpops,
    de_dir,
    batch_correction_options=(True, False),
    mode="change",
    delta=0.25,
    force=False,
):
    """Load or compute subpopulation-vs-rest DE and rankings for each batch-correction setting."""
    de_dir = Path(de_dir)
    de_by_variant = {}
    rankings_by_variant = {}
    qc_by_variant = {}
    loaded_model = None
    adata_hvg = None

    for batch_correction in batch_correction_options:
        variant = f"batch_correction_{str(batch_correction).lower()}"
        variant_dir = de_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        de_by_subpop = {}
        rankings = {}
        for subpopulation in tumor_subpops:
            de_path = variant_dir / f"scvi_de_tumor_{subpopulation}_vs_rest.csv"
            ranking_path = variant_dir / f"ranking_tumor_{subpopulation}_vs_rest.csv"
            if de_path.exists() and not force:
                print(f"loaded {variant}: {subpopulation}")
                de = pd.read_csv(de_path)
            else:
                print(f"running {variant}: {subpopulation}")
                if loaded_model is None:
                    loaded_model, adata_hvg = _ensure_model_and_adata(model, adata)
                de = subpopulation_de(
                    loaded_model,
                    adata_hvg,
                    groups_de,
                    subpopulation,
                    batch_correction=batch_correction,
                    mode=mode,
                    delta=delta,
                )
                de.to_csv(de_path, index=False)
            if ranking_path.exists() and not force:
                ranking = pd.read_csv(ranking_path)
            else:
                ranking = make_lfc_ranking(de)
                ranking.to_csv(ranking_path, index=False)
            de_by_subpop[subpopulation] = de
            rankings[subpopulation] = ranking

        qc = de_qc(de_by_subpop)
        qc.to_csv(variant_dir / "de_qc.csv", index=False)
        (variant_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "batch_correction": batch_correction,
                    "signature_mode": mode,
                    "signature_delta": delta,
                    "n_subpopulations": len(tumor_subpops),
                    "subpopulations": list(tumor_subpops),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        de_by_variant[batch_correction] = de_by_subpop
        rankings_by_variant[batch_correction] = rankings
        qc_by_variant[batch_correction] = qc

    return de_by_variant, rankings_by_variant, qc_by_variant


def load_or_run_tumor_gsea(
    rankings,
    tumor_subpops,
    gsea_dir,
    gmt_path,
    hallmark_sets,
    seed=0,
    permutations=1000,
    min_size=15,
    max_size=500,
    force=False,
):
    """Load or compute preranked Hallmark GSEA across all tumor subpopulations."""
    gsea_dir = Path(gsea_dir)
    gsea_dir.mkdir(parents=True, exist_ok=True)
    out_path = gsea_dir / "hallmark_prerank_tumor.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    results = []
    for subpopulation in tumor_subpops:
        result = run_hallmark_prerank(
            rankings[subpopulation],
            f"{subpopulation}-vs-rest",
            gmt_path,
            hallmark_sets,
            seed=seed,
            permutations=permutations,
            min_size=min_size,
            max_size=max_size,
        )
        result.insert(0, "subpopulation", subpopulation)
        results.append(result)
    results_gsea_tumor = pd.concat(results, ignore_index=True)
    results_gsea_tumor.to_csv(out_path, index=False)
    return results_gsea_tumor


def hallmark_direction_summary(results_gsea_tumor, de_qc_df, tumor_subpops, fdr=0.05):
    """Summarize FDR-significant enriched and depleted Hallmark terms per subpopulation."""
    group_col = "subpopulation" if "subpopulation" in de_qc_df.columns else de_qc_df.columns[0]
    count_col = "n_source" if "n_source" in de_qc_df.columns else "n_subpop"
    ncells_by_subpop = de_qc_df.set_index(group_col)[count_col].to_dict()

    rows = []
    for subpop in tumor_subpops:
        res = results_gsea_tumor.loc[results_gsea_tumor["subpopulation"].eq(subpop)]
        sig = res.loc[res["fdr_qval"] <= fdr]
        rows.append(
            {
                "subpopulation": subpop,
                "ncells_subpop": int(ncells_by_subpop[subpop]),
                "n_sig_fdr": int(len(sig)),
                "n_enriched": int((sig["nes"] > 0).sum()),
                "n_depleted": int((sig["nes"] < 0).sum()),
                "top_enriched": ", ".join(sig.loc[sig["nes"] > 0].nlargest(3, "nes")["pathway"]),
                "top_depleted": ", ".join(sig.loc[sig["nes"] < 0].nsmallest(3, "nes")["pathway"]),
            }
        )
    return pd.DataFrame(rows)


def load_or_run_pairwise_contrast(
    model,
    adata_hvg,
    groups_de,
    group1,
    group2,
    de_dir,
    gsea_dir,
    gmt_path,
    hallmark_sets,
    batch_correction=True,
    mode="change",
    delta=0.25,
    seed=0,
    permutations=1000,
    min_size=15,
    max_size=500,
    force=False,
):
    """Load or compute a direct head-to-head DE and Hallmark GSEA contrast between two tumor groups."""
    de_dir = Path(de_dir)
    gsea_dir = Path(gsea_dir)
    de_path = de_dir / f"scvi_de_{group1}_vs_{group2}.csv"
    gsea_path = gsea_dir / f"hallmark_prerank_{group1}_vs_{group2}.csv"

    if de_path.exists() and not force:
        pairwise_de = pd.read_csv(de_path)
    else:
        loaded_model, hvg_data = _ensure_model_and_adata(model, adata_hvg)
        pairwise_de = contrast_de(
            loaded_model,
            hvg_data,
            groups_de,
            group1,
            group2,
            batch_correction=batch_correction,
            mode=mode,
            delta=delta,
        )
        pairwise_de.to_csv(de_path, index=False)

    if gsea_path.exists() and not force:
        pairwise_gsea = pd.read_csv(gsea_path)
    else:
        pairwise_ranking = make_lfc_ranking(pairwise_de)
        pairwise_gsea = run_hallmark_prerank(
            pairwise_ranking,
            f"{group1}-vs-{group2}",
            gmt_path,
            hallmark_sets,
            seed=seed,
            permutations=permutations,
            min_size=min_size,
            max_size=max_size,
        )
        pairwise_gsea.to_csv(gsea_path, index=False)

    return pairwise_de, pairwise_gsea


def pairwise_hallmark_summary(pairwise_gsea, group1="tumor_4", group2="tumor_5", fdr=0.05):
    """Filter a pairwise GSEA table to FDR-significant pathways and label their direction."""
    sig = pairwise_gsea.loc[pairwise_gsea["fdr_qval"] <= fdr].copy()
    sig["direction"] = np.where(sig["nes"] > 0, f"up in {group1}", f"up in {group2}")
    cols = [
        "direction",
        "pathway",
        "nes",
        "nom_pval",
        "fdr_qval",
        "overlap",
        "set_size",
        "lead_genes",
    ]
    return sig.sort_values("nes", ascending=False)[cols].reset_index(drop=True)


def load_or_run_line_vs_tumor_de(
    model,
    adata_hvg,
    groups_de,
    line_groups,
    tumor_subpops,
    line_vs_tumor_dir,
    model_label="config_10_joint",
    batch_correction=True,
    mode="change",
    delta=0.25,
    force=False,
):
    """Load or compute direct cell-line versus tumor-subpopulation DE and rankings."""
    line_vs_tumor_dir = Path(line_vs_tumor_dir)
    line_vs_tumor_dir.mkdir(parents=True, exist_ok=True)
    line_de = {}
    line_rankings = {}
    loaded_model = None
    hvg_data = None

    for cell_line in line_groups:
        for subpopulation in tumor_subpops:
            pair = f"{cell_line}_vs_{subpopulation}"
            de_path = line_vs_tumor_dir / f"scvi_de_{pair}.csv"
            ranking_path = line_vs_tumor_dir / f"ranking_{pair}.csv"
            if de_path.exists() and not force:
                de = pd.read_csv(de_path)
            else:
                if loaded_model is None:
                    loaded_model, hvg_data = _ensure_model_and_adata(model, adata_hvg)
                de = line_vs_subpopulation_de(
                    loaded_model,
                    hvg_data,
                    groups_de,
                    cell_line,
                    subpopulation,
                    batch_correction=batch_correction,
                    mode=mode,
                    delta=delta,
                )
                de.to_csv(de_path, index=False)
            if ranking_path.exists() and not force:
                ranking = pd.read_csv(ranking_path)
            else:
                ranking = make_lfc_ranking(de)
                ranking.to_csv(ranking_path, index=False)
            line_de[(cell_line, subpopulation)] = de
            line_rankings[(cell_line, subpopulation)] = ranking

    qc = de_qc(
        {f"{line}_vs_{subpop}": de for (line, subpop), de in line_de.items()},
        group_name="comparison",
    )
    qc.to_csv(line_vs_tumor_dir / "line_vs_tumor_de_qc.csv", index=False)
    (line_vs_tumor_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model": model_label,
                "batch_correction": batch_correction,
                "signature_mode": mode,
                "signature_delta": delta,
                "direction": "lfc_mean > 0 means up in the cell line relative to the tumor subpopulation",
                "n_comparisons": len(line_de),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return line_de, line_rankings, qc


def load_or_run_line_vs_tumor_gsea(
    line_rankings,
    line_vs_tumor_dir,
    gmt_path,
    hallmark_sets,
    seed=0,
    permutations=1000,
    min_size=15,
    max_size=500,
    force=False,
):
    """Load or compute preranked Hallmark GSEA across all cell-line vs tumor-subpopulation pairs."""
    line_vs_tumor_dir = Path(line_vs_tumor_dir)
    out_path = line_vs_tumor_dir / "hallmark_prerank_line_vs_tumor.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    gsea_results = []
    for (cell_line, subpopulation), ranking in line_rankings.items():
        result = run_hallmark_prerank(
            ranking,
            f"{cell_line}-vs-{subpopulation}",
            gmt_path,
            hallmark_sets,
            seed=seed,
            permutations=permutations,
            min_size=min_size,
            max_size=max_size,
        )
        result.insert(0, "cell_line", str(cell_line).replace("line_", ""))
        result.insert(1, "subpop", subpopulation)
        gsea_results.append(result)
    results_gsea_line_vs_tumor = pd.concat(gsea_results, ignore_index=True)
    results_gsea_line_vs_tumor.to_csv(out_path, index=False)
    return results_gsea_line_vs_tumor


def shared_proliferation_pathways(
    results_gsea_tumor,
    proliferative=("tumor_4", "tumor_5", "tumor_12"),
    fdr=0.05,
    min_abs_nes=1.0,
):
    """Hallmark pathways significantly positively enriched in all proliferative subpopulations."""
    proliferative = list(proliferative)
    subset = results_gsea_tumor[
        results_gsea_tumor["subpopulation"].isin(proliferative)
        & results_gsea_tumor["fdr_qval"].le(fdr)
        & results_gsea_tumor["nes"].abs().ge(min_abs_nes)
    ]
    return sorted(
        pathway
        for pathway, values in subset.groupby("pathway")["nes"]
        if len(values) == len(proliferative) and bool((values > 0).all())
    )


def signature_matrix(line_signature, tumor_signature, line_order, tumor_order, exclude=()):
    """Spearman correlation matrix between cell line and tumor signatures."""
    pathways = [
        pathway
        for pathway in line_signature.columns
        if pathway in set(tumor_signature.index) and pathway not in set(exclude)
    ]
    line = line_signature.loc[line_order, pathways]
    tumor = tumor_signature.loc[pathways, tumor_order]
    return pd.DataFrame(
        {
            cell_line: {
                subpopulation: line.loc[cell_line].corr(
                    tumor.loc[:, subpopulation], method="spearman"
                )
                for subpopulation in tumor_order
            }
            for cell_line in line_order
        }
    ).T


def signature_summary(matrix, matches):
    """Primary and runner-up ranks from a cell line x subpopulation similarity matrix."""
    rows = []
    for cell_line in matrix.index:
        match_key = cell_line if cell_line in matches.index else f"line_{cell_line}"
        order = matrix.loc[cell_line].sort_values(ascending=False)
        primary = str(matches.loc[match_key, "primary_match"])
        runner_up = str(matches.loc[match_key, "second_match"])
        rows.append(
            {
                "cell_line": cell_line,
                "primary_match": primary,
                "runner_up": runner_up,
                "best_match": str(order.index[0]),
                "primary_rank": int(order.index.get_loc(primary)) + 1,
                "runner_up_rank": int(order.index.get_loc(runner_up)) + 1,
                "primary_top1": order.index[0] == primary,
                "primary_top3": int(order.index.get_loc(primary)) < 3,
                "n_subpopulations": len(order),
            }
        )
    return pd.DataFrame(rows)


def signature_readout_and_null(signature_sets, matches, n_perm=1000, seed=0):
    """Build the combined signature readout table and permutation null test summary."""
    first_matrix = next(iter(signature_sets.values()))
    line_labels = list(first_matrix.index)
    n_subpops = first_matrix.shape[1]

    readout_rows = []
    for cell_line in line_labels:
        match_key = cell_line if cell_line in matches.index else f"line_{cell_line}"
        primary = str(matches.loc[match_key, "primary_match"])
        runner_up = str(matches.loc[match_key, "second_match"])
        record = {
            "cell_line": cell_line,
            "latent_primary": primary,
            "latent_runner_up": runner_up,
        }
        for name, similarity in signature_sets.items():
            order = similarity.loc[cell_line].sort_values(ascending=False)
            record[f"{name}: best"] = str(order.index[0])
            record[f"{name}: primary rank"] = int(order.index.get_loc(primary)) + 1
            record[f"{name}: runner-up rank"] = int(order.index.get_loc(runner_up)) + 1
        readout_rows.append(record)
    readout = pd.DataFrame(readout_rows)

    rng = np.random.default_rng(seed)
    null = rng.integers(1, n_subpops + 1, size=(n_perm, len(line_labels)))
    null_rows = []
    for name in signature_sets:
        observed = readout[f"{name}: primary rank"].to_numpy()
        p_value = float((np.median(null, axis=1) <= np.median(observed)).mean())
        null_rows.append(
            {
                "signature_set": name,
                "median_primary_rank": float(np.median(observed)),
                "null_median_rank": 0.5 * (n_subpops + 1),
                "top1_rate": float((observed == 1).mean()),
                "top3_rate": float((observed <= 3).mean()),
                "permutation_p": p_value,
            }
        )
    return readout, pd.DataFrame(null_rows)
