import time
import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
from sklearn.neighbors import NearestNeighbors
from scvi.train._callbacks import SubSampleLabels

try:
    import hpo_pp_utils as ppu
except ModuleNotFoundError:
    from src import hpo_pp_utils as ppu


APPROACHES = ["joint", "scarches"]

def sample_configs(hpo_config, hvg_grid, n_configs_per_hvg, seed):
    hvg_grid = list(hvg_grid if hvg_grid is not None else hpo_config["hvg_grid"])
    n_per_hvg = int(
        n_configs_per_hvg
        if n_configs_per_hvg is not None
        else hpo_config["n_configs_per_hvg"]
    )
    seed = int(seed if seed is not None else hpo_config["seed"])
    
    search_space = hpo_config["search_space"]
    fixed = hpo_config["fixed"]

    keys = list(search_space.keys())
    grid = list(itertools.product(*[search_space[k] for k in keys]))
    n_total = len(grid)
    n_configs = n_per_hvg * len(hvg_grid)

    if n_configs > n_total:
        raise ValueError(
            f"{n_per_hvg} configs per HVG * {len(hvg_grid)} HVG levels "
            f"exceeds the {n_total} unique hyperparameter combinations."
        )
    
    # Draw all configs without replacement then split them up across the HVGs
    rng = np.random.RandomState(seed)
    chosen = rng.choice(n_total, size=n_configs, replace=False)

    configs = []
    for config_id, i in enumerate(chosen):
        num_hvg = hvg_grid[config_id // n_per_hvg]
        
        cfg = {k: v for k, v in zip(keys, grid[i])}
        cfg.update(dict(fixed))
        cfg["config_id"] = config_id
        cfg["num_hvg"] = num_hvg
        cfg["seed"] = seed
        cfg["grid_index"] = int(i)
        configs.append(cfg)
        
    return configs

# ---------------------------------------------------------------------------
# Run bookkeeping
# ---------------------------------------------------------------------------
def is_run_done(run_dir):
    """True only when the run finished successfully (DONE marker exists).

    There are only two states: done or not done. Failed or interrupted runs
    have no DONE marker and are retried on the next execution; the error
    message is recorded in eval_summary.json for debugging.
    """
    return (Path(run_dir) / "DONE").exists()


def mark_run_done(run_dir):
    (Path(run_dir) / "DONE").touch()


def mark_run_error(run_dir, message):
    """Record a failed run's error message in eval_summary.json.

    The run keeps no DONE marker, so it is retried on the next execution.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval_summary.json").write_text(
        json.dumps({"error": str(message)}, indent=2)
    )


def set_seed(seed):
    scvi.settings.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)


def json_safe(obj):
    """Recursively convert numpy scalars/NaN so json.dumps does not fail."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def json_dumps(obj):
    return json.dumps(json_safe(obj), indent=2)


def append_results_row(results_path, summary):
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([summary])
    if results_path.exists():
        existing = pd.read_csv(results_path)
        keep = ~(
            (existing["config_id"].astype(int) == int(summary["config_id"]))
            & (existing["approach"] == summary["approach"])
        )
        row = pd.concat([existing[keep], row], ignore_index=True)
    row.to_csv(results_path, index=False)


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------
def scvi_model_kwargs(config):
    return dict(
        n_latent=config["n_latent"],
        n_hidden=config["n_hidden"],
        n_layers=config["n_layers"],
        dispersion=config["dispersion"],
        encode_covariates=config["encode_covariates"],

        gene_likelihood=config["gene_likelihood"],
        use_observed_lib_size=config["use_observed_lib_size"],
        deeply_inject_covariates=config["deeply_inject_covariates"],
    )


def training_kwargs(config, warmup, external_indexing, n_samples_per_label=None,
                    load_best_on_end=True):
    callbacks = [
        scvi.train.SaveCheckpoint(
            monitor="elbo_validation", load_best_on_end=load_best_on_end
        )
    ]
    # scvi-tools 1.4.2 omits its default sampler callback when custom
    # callbacks are supplied, so preserve per-epoch label resampling here.
    if n_samples_per_label is not None:
        callbacks.append(SubSampleLabels())

    kwargs = dict(
        max_epochs=config["max_epochs"],
        batch_size=config["batch_size"],
        early_stopping=config["early_stopping"],
        early_stopping_patience=config["early_stopping_patience"],
        plan_kwargs={"n_epochs_kl_warmup": warmup, "lr": config["lr"]},
        datasplitter_kwargs={"external_indexing": list(external_indexing), "num_workers": 0},
        callbacks=callbacks,
    )
    kwargs["early_stopping_warmup_epochs"] = warmup
    # SCANVI.train accepts n_samples_per_label directly (scvi-tools 1.4.x);
    # passing it inside datasplitter_kwargs would collide with the explicit
    # argument and raise a TypeError.
    if n_samples_per_label is not None:
        kwargs["n_samples_per_label"] = n_samples_per_label
    
    return kwargs


def fit_scanvi(adata, config):
    warmup = config["n_epochs_kl_warmup"]["joint"] # fit always on joint kl warmup
    
    scvi.model.SCVI.setup_anndata(
        adata,
        layer="counts",
        batch_key=config["batch_key"],
        labels_key=config["labels_key"],
    )
    train_idx, val_idx = ppu.stratified_train_val_indices(
        adata, config["batch_key"], config["labels_key"], seed=config["seed"]
    )

    scvi_model = scvi.model.SCVI(adata, **scvi_model_kwargs(config))
    scvi_model.train(
        **training_kwargs(config, warmup, [train_idx, val_idx])
    )

    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model, labels_key=config["labels_key"],
        unlabeled_category=config["unlabeled_category"]
    )
    scanvi_model.train(
        **training_kwargs(
            config, warmup, [train_idx, val_idx],
            n_samples_per_label=config["n_samples_per_label"]
        )
    )
    return scanvi_model

# ---------------------------------------------------------------------------
# Approach training
# ---------------------------------------------------------------------------
def train_joint_scanvi(combined, config, run_dir):
    model = fit_scanvi(combined, config)
    model.save(run_dir / "joint_scanvi", overwrite=True)
    return model


def train_scarches(tumor, cl, config, run_dir):
    ref_model = fit_scanvi(tumor, config)
    ref_dir = run_dir / "reference"
    ref_model.save(ref_dir, overwrite=True)
    
    query_model = scvi.model.SCANVI.load_query_data(
        cl, reference_model=ref_model
    )
    train_idx, val_idx = ppu.stratified_train_val_indices(
        cl, config["batch_key"], config["labels_key"], seed=config["seed"]
    )
    query_model.train(
        **training_kwargs(
            config, config["n_epochs_kl_warmup"]["scarches"],
            [train_idx, val_idx],
            #n_samples_per_label=config["n_samples_per_label"] #CL is unlabeled
            # The query's validation ELBO follows the KL-warmup schedule, so
            # the min-ELBO checkpoint is a mid-warmup artifact; keep the
            # final (post-warmup) state instead of reloading a "best".
            load_best_on_end=False,
        )
    )
    query_dir = run_dir / "query"
    query_model.save(query_dir, overwrite=True)
    
    return ref_model, query_model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def class_aware_subset_indices(adata, n_cells, seed, label_key="label"):
    """Class-aware scIB subset: classes <= fair share fully included, larger
    classes downsampled proportionally to exactly n_cells (largest remainder).

    Returns positional indices into `adata`: all cell-line cells + `n_cells`
    class-balanced tumor cells.
    """
    mask = adata.obs["dataset"] == "tumor"
    tumor = adata[mask]
    cl = adata[~mask]
    n_cl = cl.n_obs

    group_values = np.asarray(tumor.obs[label_key])
    n_cells = int(n_cells)
    n_total = len(group_values)

    if n_cells <= 0:
        return np.arange(n_cl)
    if n_cells >= n_total:
        return np.concatenate([np.arange(n_total) + n_cl, np.arange(n_cl)])

    n_classes = len(np.unique(group_values))
    fair_share = max(1, n_cells // n_classes)
    rng = np.random.RandomState(seed)

    small, large = [], []
    for g in np.unique(group_values):
        cells = np.where(group_values == g)[0]
        (small if len(cells) <= fair_share else large).append(cells)

    idx = list(small)
    budget = n_cells - sum(len(c) for c in small)
    if budget > 0 and large:
        sizes = np.array([len(c) for c in large], dtype=float)
        shares = budget * sizes / sizes.sum()
        alloc = np.floor(shares).astype(int)
        leftover = budget - alloc.sum()
        order = np.argsort(shares - alloc)[::-1]
        for j in range(leftover):
            alloc[order[j % len(order)]] += 1
        for cells, n in zip(large, alloc):
            idx.append(rng.choice(cells, size=int(n), replace=False))

    idx = np.concatenate(idx) + n_cl  # shift: all cl cells + ncells tumor
    return np.concatenate([idx, np.arange(n_cl)])


def run_scib(config, combined, latent, scib_n_cells):
    """Benchmark only on scib_n_cells + cl.n_obs cells for faster computation.

    Two benchmarks on complementary subsets of the same class-aware selection:
    - Labeled benchmark: BioConservation metrics (isolated labels, NMI/ARI
      kmeans, silhouette label, cLISI) plus the label-conditioned batch
      metrics (BRAS, kBET per label, graph connectivity) on the fully labeled
      tumor cells. These metrics are computed *within* label groups; running
      them on cl+tumor data would treat all cell lines as one "Unknown" label
      and measure mixing/connectivity inside that meaningless group.
    - Full benchmark: the label-free batch metrics (iLISI, PCR comparison) on
      the same subset plus all cell-line cells (mixing across donors and
      pools).

    Returns the two score Series without scib-metrics' aggregate rows (they
    are recomputed per metric type in `summarize_eval`, see there).

    scib-metrics 0.5.9 returns the benchmarked embeddings as DataFrame rows
    (index "Embedding"), so the embedding key is selected by transposing
    (`.T["X_scanvi"]`). The default `clean_names=True` keeps the readable
    metric names ("Isolated labels", "KMeans NMI", ...).
    """
    from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
    
    n_tumor = int((combined.obs["dataset"] == "tumor").sum())
    if scib_n_cells is None or scib_n_cells <= 0 or scib_n_cells >= n_tumor:
        scib_n_cells = n_tumor
    indices = class_aware_subset_indices(combined, scib_n_cells, config["seed"])
    tumor_mask = (combined.obs["dataset"] == "tumor").to_numpy()  # only tumor cells labeled
    subset_mask = np.zeros(combined.n_obs, dtype=bool)
    subset_mask[indices] = True
    tumor_subset_mask = subset_mask & tumor_mask

    # Label-conditioned metrics on the labeled tumor subset.
    tumor_subset_eval = combined[tumor_subset_mask].copy()
    tumor_subset_eval.obsm["X_scanvi"] = latent[tumor_subset_mask]
    sc.tl.pca(tumor_subset_eval)

    bio_metrics = BioConservation(
        isolated_labels=True,
        nmi_ari_cluster_labels_leiden=False,
        nmi_ari_cluster_labels_kmeans=True,
        silhouette_label=True,
        clisi_knn=True,
    )
    batch_metrics_labeled = BatchCorrection(
        bras=True,
        ilisi_knn=False,  # label-free: run on cl+tumor in the full benchmark
        kbet_per_label=True,
        graph_connectivity=True,
        pcr_comparison=False,  # label-free: run on cl+tumor in the full benchmark
    )
    bm_labeled = Benchmarker(
        adata=tumor_subset_eval,
        batch_key=config["batch_key"],
        label_key=config["labels_key"],
        embedding_obsm_keys=["X_scanvi"],
        bio_conservation_metrics=bio_metrics,
        batch_correction_metrics=batch_metrics_labeled,
        pre_integrated_embedding_obsm_key="X_pca",
        n_jobs=-1,
        progress_bar=False,
    )
    bm_labeled.benchmark()
    labeled_scores = bm_labeled.get_results(min_max_scale=False).T["X_scanvi"].drop(["Total", "Batch Correction"], errors="ignore")
    
    
    # Label-free batch metrics on the tumor subset plus all cell-line cells.
    eval_adata = combined[indices].copy()
    eval_adata.obsm["X_scanvi"] = latent[indices]
    sc.tl.pca(eval_adata)
    bm_full = Benchmarker(
        adata=eval_adata,
        batch_key=config["batch_key"],
        label_key=config["labels_key"],
        embedding_obsm_keys=["X_scanvi"],
        bio_conservation_metrics=None,
        batch_correction_metrics=BatchCorrection(
            bras=False,
            ilisi_knn=True,
            kbet_per_label=False,
            graph_connectivity=False,
            pcr_comparison=True,
        ),
        pre_integrated_embedding_obsm_key="X_pca",
        n_jobs=-1,
        progress_bar=False,
    )
    bm_full.benchmark()
    full_scores = bm_full.get_results(min_max_scale=False).T["X_scanvi"].drop(["Total", "Batch correction", "Bio conservation"], errors="ignore")
    
    return labeled_scores, full_scores


def knn_mixing_scores(config, combined, latent_full):
    """kNN mixing ratio and tumor-label entropy for every cell-line cell.

    Returns a DataFrame with one row per cell-line cell: mixing_k{k} (fraction
    of the k nearest neighbors that are tumor cells), entropy_k{k} (normalized
    tumor-label entropy of the tumor neighbors, NaN without any tumor
    neighbor), coverage_k{k} (1 if at least one tumor neighbor) and
    line_purity_k{k} (fraction of the k nearest *cell-line* neighbors sharing
    the cell's line), for k in config["k_values"].
    """
    k_values = config["k_values"]
    labels_key = config["labels_key"]
    batch_key = config["batch_key"]
    tumor_mask = (combined.obs["dataset"] == "tumor").to_numpy()
    cl_mask = ~tumor_mask
    cl_idx = np.where(cl_mask)[0]

    n_labels = combined.obs.loc[tumor_mask, labels_key].nunique()
    if n_labels < 2:
        raise ValueError("Need at least 2 tumor labels for entropy normalization.")

    nn = NearestNeighbors(n_neighbors=max(k_values) + 1, n_jobs=1)
    nn.fit(latent_full)
    _, neighbor_idx = nn.kneighbors(latent_full[cl_idx])
    # Self is always the nearest neighbor (distance 0); drop it.
    neighbor_idx = neighbor_idx[:, 1:]

    # Same-line purity: kNN restricted to cell-line cells.
    cl_nn = NearestNeighbors(n_neighbors=max(k_values) + 1, n_jobs=1)
    cl_nn.fit(latent_full[cl_idx])
    _, cl_neighbor_idx = cl_nn.kneighbors(latent_full[cl_idx])
    cl_neighbor_idx = cl_neighbor_idx[:, 1:]
    cl_cell_lines = combined.obs["cell_line"].to_numpy()[cl_idx]
    cl_neighbor_lines = cl_cell_lines[cl_neighbor_idx]

    neighbor_labels = combined.obs[labels_key].to_numpy()[neighbor_idx]

    records = []
    for row, gidx in enumerate(cl_idx):
        record = {
            "obs_index": combined.obs_names[gidx],
            "cell_line": combined.obs["cell_line"].iloc[gidx],
            "batch": combined.obs[batch_key].iloc[gidx],
        }
        for k in k_values:
            nbrs = neighbor_idx[row, :k]
            is_tumor = tumor_mask[nbrs]
            mixing = is_tumor.sum() / k
            tumor_labels = neighbor_labels[row, :k][is_tumor]
            record[f"mixing_k{k}"] = mixing
            record[f"coverage_k{k}"] = int(mixing > 0)
            if len(tumor_labels) > 0:
                counts = np.bincount(
                    pd.Categorical(tumor_labels).codes, minlength=n_labels
                )
                probs = counts / counts.sum()
                probs = probs[probs > 0]  # 0 * log(0) would be NaN
                entropy = -np.sum(probs * np.log(probs)) / np.log(n_labels)
            else:
                entropy = np.nan
            record[f"entropy_k{k}"] = entropy
            record[f"line_purity_k{k}"] = float(
                (cl_neighbor_lines[row, :k] == cl_cell_lines[row]).mean()
            )
        records.append(record)

    return pd.DataFrame(records)


def history_metric_values(history, key):
    """Return one training metric as a flat NumPy array.

    scvi-tools' SimpleLogger stores each history metric as a one-column
    pandas DataFrame, while older versions may expose a Series or array.
    """
    if history is None:
        return None
    values = history.get(key)
    if values is None:
        return None
    if isinstance(values, pd.DataFrame):
        if key in values.columns:
            values = values[key]
        elif values.shape[1] == 1:
            values = values.iloc[:, 0]
        else:
            raise ValueError(f"History metric {key!r} has unexpected columns.")
    return np.asarray(values).reshape(-1)


def saved_best_epoch(model, load_best_on_end=True):
    """Epoch (0-based) of the model state that is saved after training.

    With `load_best_on_end=True` (joint/reference), the SaveCheckpoint
    callback reloads the minimum validation ELBO (mode "min", first
    occurrence wins, NaN never beats), which is exactly the argmin over the
    validation history. With `load_best_on_end=False` (scArches query), the
    saved state is the final one, i.e. the last trained epoch. `n_epochs`
    (training length) and the saved-model epoch differ whenever the best
    validation ELBO was reached before the end of training.
    """
    history = getattr(model, "history", None)
    if history is None:
        return None
    vals = history_metric_values(history, "elbo_validation")
    if vals is None or vals.size == 0:
        return None
    if not load_best_on_end:
        return len(vals) - 1
    best_idx, best_val = None, np.inf
    for i, v in enumerate(vals):
        if pd.isna(v):
            continue
        v = float(v)
        if v < best_val:
            best_val = v
            best_idx = i
    return best_idx


def summarize_eval(config, approach, history, labeled_scores, full_scores,
                   mixing_df, train_time_s=None, scib_n_cells=None,
                   best_epoch=None):
    """Build the summary row written to results.csv/eval_summary.json.

    Bio-conservation and the label-conditioned batch metrics come from the
    labeled benchmark, the label-free batch metrics (iLISI, PCR comparison)
    from the full benchmark; everything is dumped as one metric set without
    prefixes. The aggregates are recomputed over the dumped metrics:
    "Bio conservation" = mean of the bio metrics, "Batch correction" = mean
    of all five batch metrics (labeled + full), "Total" = 0.4 x Batch
    correction + 0.6 x Bio conservation (scIB weighting across benchmarks).
    """
    k_values = config["k_values"]
    elbo_values = history_metric_values(history, "elbo_validation")
    summary = dict(
        config_id=config["config_id"],
        num_hvg=config["num_hvg"],
        approach=approach,
        n_layers=config["n_layers"],
        n_hidden=config["n_hidden"],
        n_latent=config["n_latent"],
        dispersion=config["dispersion"],
        encode_covariates=config["encode_covariates"],
        n_samples_per_label=config["n_samples_per_label"],
        n_epochs=None if elbo_values is None else len(elbo_values),
        elbo_validation=None if elbo_values is None else float(np.min(elbo_values)),
        best_epoch=best_epoch,
        train_time_s=train_time_s,
    )
    if scib_n_cells is not None:
        summary["scib_n_cells"] = int(scib_n_cells)

    for metric, value in labeled_scores.items():
        summary[metric] = float(value)
    for metric, value in full_scores.items():
        summary[metric] = float(value)

    scib_bio_metrics = [
        "Isolated labels", "KMeans NMI",
        "KMeans ARI", "Silhouette label",
        "cLISI",
    ]
    scib_batch_metrics = [
        "BRAS", "KBET", "Graph connectivity",
        "iLISI", "PCR comparison",
    ]

    # Bio conservation already included from the benchmarking
    summary["Batch correction"] = float(
        np.nanmean([summary[m] for m in scib_batch_metrics])
    )
    summary["Total"] = (
        0.4 * summary["Batch correction"] + 0.6 * summary["Bio conservation"]
    )

    for k in k_values:
        col_m = f"mixing_k{k}"
        col_e = f"entropy_k{k}"
        col_c = f"coverage_k{k}"
        col_p = f"line_purity_k{k}"
        summary[f"mixing_{k}"] = float(mixing_df[col_m].mean())
        summary[f"entropy_{k}"] = float(mixing_df[col_e].mean())
        summary[f"coverage_{k}"] = float(mixing_df[col_c].mean())
        summary[f"line_purity_{k}"] = float(mixing_df[col_p].mean())

    return summary


def summarize_line_mixing(mixing_df, k_values):
    """Per-cell-line means of the per-cell kNN mixing scores.

    One row per cell line: its cell count plus the per-k means of
    mixing/entropy/coverage/line_purity. Written alongside the per-cell
    `cell_line_mixing.csv` as `cell_line_mixing_summary.csv`; the per-cell
    file stays the raw record for per-line boxplots or other aggregations.
    """
    k_values = list(k_values)
    cols = [f"{m}_k{k}" for m in ("mixing", "entropy", "coverage", "line_purity")
            for k in k_values]
    per_line = mixing_df.groupby("cell_line", sort=False)[cols].mean()
    per_line.insert(0, "n_cells", mixing_df.groupby("cell_line", sort=False).size())
    return per_line.reset_index()


def save_run_results(run_dir, config, approach, labeled_scores, full_scores,
                     mixing_df, summary):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json_dumps(config))

    # Per-benchmark metric rows; the aggregate rows ("Batch correction" from
    # the labeled benchmark, "Bio conservation") are replaced by the
    # recomputed ones from the summary so scib_results.csv agrees with
    # eval_summary.json, including the overall "Total" score.
    scib_df = pd.concat(
        [labeled_scores.rename("labeled"), full_scores.rename("full")],
        axis=1,
    ).drop(
        ["Batch correction", "Bio conservation", "Total"], errors="ignore"
    )
    scib_df = scib_df.reset_index().rename(columns={"index": "metric"})
    aggregate_rows = pd.DataFrame(
        {
            "metric": ["Batch correction", "Bio conservation", "Total"],
            "overall": [summary[m] for m in ("Batch correction", "Bio conservation", "Total")],
        }
    )
    scib_df = pd.concat([scib_df, aggregate_rows], ignore_index=True)
    scib_df.to_csv(run_dir / "scib_results.csv", index=False)

    mixing_df.to_csv(run_dir / "cell_line_mixing.csv", index=False)
    summarize_line_mixing(mixing_df, config["k_values"]).to_csv(
        run_dir / "cell_line_mixing_summary.csv", index=False
    )

    (run_dir / "eval_summary.json").write_text(json_dumps(summary))


def evaluate(model, combined, config, run_dir, approach, scib_n_cells=None,
             train_time_s=None):
    """Run the full per-(config, approach) evaluation.

    `model` embeds all cells: for `joint` the SCANVI trained on the combined
    data; for `scarches` the query SCANVI with the combined adata passed in
    (its registry covers reference + query batches via `extend_categories`).
    The query model gives identical reference-domain latents to the reference
    model because scArches query training (default freeze flags) only updates
    the newly added batch parameters: expression weights are frozen and the
    reference-batch columns never receive gradients during query training.
    Writes config.json, scib_results.csv, cell_line_mixing.csv,
    cell_line_mixing_summary.csv and eval_summary.json into run_dir and
    returns the summary dict.
    """
    latent_full = model.get_latent_representation(combined)

    labeled_scores, full_scores = run_scib(
        config, combined, latent_full, scib_n_cells
    )
    mixing_df = knn_mixing_scores(config, combined, latent_full)
    summary = summarize_eval(
        config, approach, model.history, labeled_scores, full_scores, mixing_df,
        train_time_s=train_time_s, scib_n_cells=scib_n_cells,
        best_epoch=saved_best_epoch(model, load_best_on_end=(approach != "scarches")),
    )
    save_run_results(run_dir, config, approach, labeled_scores, full_scores,
                     mixing_df, summary)

    return summary
