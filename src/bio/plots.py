"""Reusable plots for the biological-plausibility notebooks."""

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns


def plot_leiden_sweep(adata, prefix, resolutions):
    """Plot UMAPs for a sweep of Leiden resolutions plus Pelka labels and cell lines."""
    cols = [f"{prefix}_leiden_res{res}" for res in resolutions]
    sc.pl.umap(adata, color=cols, legend_loc="on data")
    sc.pl.umap(adata, color="label")
    sc.pl.umap(
        adata,
        mask_obs=adata.obs["dataset"].eq("cell_line"),
        color=["cell_line"],
        na_in_legend=False,
        size=10,
        alpha=0.45,
    )
