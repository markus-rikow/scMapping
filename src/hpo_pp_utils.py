import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

from sklearn.model_selection import train_test_split

def compute_combined_hvg(tumor, cl, n_hvg, flavor="seurat_v3", batch_key="dataset"):
    combined = ad.concat([cl, tumor])

    sc.pp.highly_variable_genes(
        combined,
        n_top_genes=n_hvg,
        subset=True,
        layer=None, # we use .X as we did not yet transfer counts to layers
        flavor=flavor,
        batch_key=batch_key,
    )
    hvgs = combined.var_names[combined.var["highly_variable"]]
    del combined
    return hvgs


def build_combined_adata(tumor, cl):
    combined = ad.concat([cl, tumor])

    # the caller normally transfers raw counts to layers["counts"] on tumor/cl
    # beforehand; fall back to X if not done yet
    if "counts" not in combined.layers:
        combined.layers["counts"] = combined.X.copy()
    sc.pp.normalize_total(combined, target_sum=1e4)
    sc.pp.log1p(combined)

    return combined


def stratified_train_val_indices(adata, batch_key, labels_key, val_fraction=0.1, seed=0):
    # batch x label groups to stratify tumor
    # batch x cell line groups to stratify cell lines
    batch_vals = adata.obs[batch_key].astype(str)
    labels_val = adata.obs[labels_key].astype(str)
    cl_vals = adata.obs["cell_line"].astype(str)
    is_numeric_batch = batch_vals.str.isnumeric()
    
    groups = pd.Series(
        np.where(
            is_numeric_batch,
            batch_vals + "_" + cl_vals,
            batch_vals + "_" + labels_val
        )
    )
    counts = groups.value_counts()
    
    # groups with just 1 cell assigned are always in train
    rare_mask = groups.isin(counts[counts < 2].index)
    rare_idx = np.where(rare_mask)[0]

    # remaining valid groups can be split
    valid_mask = ~rare_mask
    valid_idx  = np.where(valid_mask)[0]
    valid_groups = groups[valid_mask]
    train_valid, val_idx = train_test_split(
        valid_idx, test_size=val_fraction, stratify=valid_groups, random_state=seed
    )

    train_idx = np.concatenate([rare_idx, train_valid])
    return train_idx, val_idx