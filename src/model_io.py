"""Model loading and latent reconstruction for saved HPO runs.

The HPO sweep does not store latent representations — only the trained model
(``joint_scanvi/`` or ``query/``) plus ``config.json`` and the HVG list are
kept. This module reloads a saved run and recomputes its combined latent on
the rebuilt (cell lines + tumor) HVG adata, so post-HPO analyses (ranking,
biological validation) see exactly the latent the run was evaluated with.

- ``load_run_latent(run_dir, adata_tumor, adata_cl, approach)`` is the
  all-in-one entry point: it reads the run's config, rebuilds the combined
  adata internally, loads the model and returns the latent, the obs arrays
  and the rebuilt combined adata.
- ``build_combined_adata_for_hvg`` and ``load_latent_from_combined`` expose
  the two halves for callers that want to cache the combined adata across
  runs of the same HVG level (e.g. the post-HPO ranking over all 102 runs).

Row conventions: ``dataset`` marks tumor vs cell-line
rows, cell-line rows carry ``cell_line``/``batch``, tumor rows carry
``label``.
"""

import json
from pathlib import Path

import numpy as np

from src import hpo_pp_utils as ppu


def build_combined_adata_for_hvg(adata_tumor, adata_cl, num_hvg, hvg_dir):
    """Rebuild the combined HVG adata a run was trained on.

    Mirrors `scripts/04_hpo.py`: subset both datasets to the run's HVG list,
    move the raw counts into ``layers["counts"]``, then
    ``hpo_pp_utils.build_combined_adata`` (normalize_total(1e4) + log1p in
    ``X``). ``hvg_dir`` is the ``hvg<num_hvg>`` directory holding
    ``hvg_genes.txt``.
    """
    hvg_dir = Path(hvg_dir)
    hvg_genes = (hvg_dir / "hvg_genes.txt").read_text().splitlines()
    tumor_hvg = adata_tumor[:, hvg_genes].copy()
    cl_hvg = adata_cl[:, hvg_genes].copy()
    tumor_hvg.layers["counts"] = tumor_hvg.X.copy()
    cl_hvg.layers["counts"] = cl_hvg.X.copy()
    return ppu.build_combined_adata(tumor_hvg, cl_hvg)


def load_latent_from_combined(run_dir, combined, approach):
    """Load the saved model on ``combined`` and return the latent + obs.

    ``run_dir`` holds the saved model (``joint_scanvi`` for "joint",
    ``query`` for "scarches"; the query model's registry covers the reference
    + pool batches, so the combined adata can be passed to it directly).
    Returns the same dict shape the old ``latent.npz`` files provided
    (latent, obs_names, dataset, cell_line, label, batch).
    """
    import scvi

    run_dir = Path(run_dir)
    if approach == "joint":
        model = scvi.model.SCANVI.load(
            str(run_dir / "joint_scanvi"), adata=combined
        )
    elif approach == "scarches":
        model = scvi.model.SCANVI.load(
            str(run_dir / "query"), adata=combined
        )
    else:
        raise ValueError(f"Unknown approach: {approach}")

    latent = np.asarray(model.get_latent_representation(combined))
    return {
        "latent": latent,
        "obs_names": np.asarray(combined.obs_names),
        "dataset": np.asarray(combined.obs["dataset"]),
        "cell_line": np.asarray(combined.obs["cell_line"]),
        "label": np.asarray(combined.obs["label"]),
        "batch": np.asarray(combined.obs["batch"]),
    }


def load_run_latent(run_dir, adata_tumor, adata_cl, approach):
    """All-in-one: rebuild the run's combined adata and recompute the latent.

    Reads ``num_hvg`` from the run's ``config.json``, rebuilds the combined
    HVG adata internally, then loads the model. Returns the
    :func:`load_latent_from_combined` dict plus ``combined`` (the rebuilt
    adata) and ``num_hvg``.
    """
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    num_hvg = int(config["num_hvg"])
    combined = build_combined_adata_for_hvg(
        adata_tumor, adata_cl, num_hvg, hvg_dir=run_dir.parents[1]
    )
    out = load_latent_from_combined(run_dir, combined, approach)
    out["combined"] = combined
    out["num_hvg"] = num_hvg
    return out
