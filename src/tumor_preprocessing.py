import h5py
import anndata as ad
import pandas as pd
import numpy as np

from scipy import sparse


def load_tumor_h5_as_adata(h5_file_path):
    data = h5py.File(h5_file_path, "r")

    matrix_group = data["matrix"]
    counts = matrix_group["data"][:]
    indices = matrix_group["indices"][:]
    indptr = matrix_group["indptr"][:]
    shape = tuple(matrix_group["shape"][:])

    features = matrix_group["features"]
    gene_ids = [x.decode() if isinstance(x, bytes) else x for x in features["id"][:]]
    gene_names = [x.decode() if isinstance(x, bytes) else x for x in features["name"][:]]
    cell_barcodes = [x.decode() if isinstance(x, bytes) else x for x in matrix_group["barcodes"][:]]

    mat = sparse.csc_matrix((counts, indices, indptr), shape=shape, dtype="uint32")
    obs = pd.DataFrame(index=cell_barcodes)
    var_id = pd.DataFrame(index=gene_ids) # gene_symbol not unique; Using id with geneAnno later
    adata = ad.AnnData(X=mat.T.tocsr(), obs=obs, var=var_id)
    
    assert adata.shape == (len(cell_barcodes), len(gene_ids))

    adata.var["gene_symbol"] = gene_names
    adata.var["gene_id"] = adata.var.index
    adata.obs["cell_id"] = adata.obs.index
    adata.obs["donor_id"] = adata.obs["cell_id"].str.extract(r'^(C\d{2,4})')
    adata.var["gene_id"] = adata.var["gene_id"].str.split(".").str[0]
    
    assert adata.obs["donor_id"].notna().all()
    return adata


def attach_metadata(adata, metadata_path):
    metadata = pd.read_csv(
        metadata_path,
        sep="\t",
        usecols=["NAME", "donor_id", "MMRStatusTumor", "library_preparation_protocol__ontology_label", "biosample_id"]
    ).iloc[1:]
    
    metadata = metadata.rename(columns = {"NAME":"cell_id", "library_preparation_protocol__ontology_label": "chemistry", "MMRStatusTumor":"mmr_status"})

    metadata["chemistry"] = np.where(
        metadata["chemistry"].str.contains("v2", na=False),
        "v2",
        np.where(
            metadata["chemistry"].str.contains("v3", na=False),
            "v3",
            metadata["chemistry"]
        )
    )

    new_obs = adata.obs.merge(metadata, on=["cell_id", "donor_id"], how="inner")
    new_obs.index = new_obs["cell_id"]
    new_obs.index.name = None
    adata.obs = new_obs
    
    return adata


def attach_cell_state_label(adata, label_path, batch_style="donor_chemistry"):
    labels = pd.read_csv(label_path, sep="\t", usecols=["NAME", "ClusterFull", "ClusterMidway"]).iloc[1:]
    labels = labels.rename(columns = {"NAME":"cell_id", "ClusterFull":"label", "ClusterMidway":"cell_type"})
    extracted = adata.obs['biosample_id'].str.extract(r'^(?P<donor>C\d+)_(?P<sample>[A-Z]+)_.*_(?P<chemistry>v\d+)$')

    if batch_style == "donor_chemistry":
        # C130/C171 with two lesions (TA/TB) are treated as the same donor biology,
        # matching the batch design used by the HPO.
        def make_batch(row):
            return f"{row['donor']}_{row['chemistry']}"
    elif batch_style == "donor_sample_chemistry":
        # keep lesions of C130/C171 as separate batches: As the first idea but should be discard
        # according to Luecken scBestPracties-Book: Data Integration (Which batch label box)
        def make_batch(row):
            if row["sample"] == "T":
                return f"{row['donor']}_{row['chemistry']}"
            else:
                return f"{row['donor']}_{row['sample']}_{row['chemistry']}"
    else:
        raise ValueError(f"Unknown batch_style: {batch_style}. Use 'donor_chemistry' or 'donor_sample_chemistry'.")

    # add batch_key and cell state label
    adata.obs["batch"] = extracted.apply(make_batch, axis=1)
    new_obs = adata.obs.merge(labels, on="cell_id", how="inner")
    new_obs.index = new_obs["cell_id"].astype(str)
    new_obs.index.name = None
    adata.obs = new_obs

    adata.obs["dataset"] = "tumor"
    adata.obs["cell_line"] = "tumor"

    return adata


def attach_paper_extra_metadata(adata):
    donor_metadata = {
        "C106":  {"MMR_status": "MMRd", "mlh1_status": True, "sex": "male", "tissue_site": "right", "pathological_stage": 4, "tumor_grade": "low"},
        "C109":  {"MMR_status": "MMRd", "mlh1_status": True, "sex": "female", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "low"},
        "C110":  {"MMR_status": "MMRd", "mlh1_status": True, "sex": "male", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "high"},
        "C111":  {"MMR_status": "MMRd", "mlh1_status": True, "sex": "female", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "low"},
        "C114":  {"MMR_status": "MMRd", "mlh1_status": True, "sex": "male", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "low"},
        "C115":  {"MMR_status": "MMRd", "mlh1_status": False, "sex": "male", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "low"},
        "C116":  {"MMR_status": "MMRd", "mlh1_status": False, "sex": "male", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "low"},
        "C118":  {"MMR_status": "MMRd", "mlh1_status": True, "sex": "male", "tissue_site": "right", "pathological_stage": 2, "tumor_grade": "low"},
        "C119":  {"MMR_status": "MMRd", "mlh1_status": False, "sex": "male", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "high"},
        "C122":  {"MMR_status": "MMRd", "mlh1_status": True, "sex": "female", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "low"},
        "C123":  {"MMR_status": "MMRd", "mlh1_status": True, "sex": "male", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "low"},
        "C130":  {"MMR_status": "MMRd", "mlh1_status": True, "sex": "female", "tissue_site": "right", "pathological_stage": 4, "tumor_grade": "low"},
        "C132":  {"MMR_status": "MMRd", "mlh1_status": False, "sex": "female", "tissue_site": "left", "pathological_stage": 4, "tumor_grade": "high"},
        "C137":  {"MMR_status": "MMRd", "mlh1_status": False, "sex": "female", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "high"},
        "C138":  {"MMR_status": "MMRd", "mlh1_status": False, "sex": "male", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "low"},
        "C139":  {"MMR_status": "MMRd", "mlh1_status": True, "sex": "female", "tissue_site": "right", "pathological_stage": 4, "tumor_grade": "low"},
        "C142":  {"MMR_status": "MMRd", "mlh1_status": False, "sex": "female", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "low"},
        "C143":  {"MMR_status": "MMRd", "mlh1_status": False, "sex": "female", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "low"},
        "C144":  {"MMR_status": "MMRd", "mlh1_status": True, "sex": "male", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "low"},
        "C146":  {"MMR_status": "MMRd", "mlh1_status": True, "sex": "female", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "low"},
        "C147":  {"MMR_status": "MMRd", "mlh1_status": True, "sex": "female", "tissue_site": "right", "pathological_stage": 4, "tumor_grade": "low"},
        "C151":  {"MMR_status": "MMRd", "mlh1_status": True, "sex": "female", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "high"},
        "C152":  {"MMR_status": "MMRd", "mlh1_status": True, "sex": "female", "tissue_site": "right", "pathological_stage": 2, "tumor_grade": "low"},
        "C154":  {"MMR_status": "MMRd", "mlh1_status": True, "sex": "female", "tissue_site": "right", "pathological_stage": 2, "tumor_grade": "low"},
        "C156":  {"MMR_status": "MMRd", "mlh1_status": True, "sex": "female", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "low"},
        "C158":  {"MMR_status": "MMRd", "mlh1_status": True, "sex": "female", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "low"},
        "C163":  {"MMR_status": "MMRd", "mlh1_status": False, "sex": "female", "tissue_site": "right", "pathological_stage": 4, "tumor_grade": "high"},
        "C164":  {"MMR_status": "MMRd", "mlh1_status": False, "sex": "female", "tissue_site": "right", "pathological_stage": 4, "tumor_grade": "low"},
        "C165":  {"MMR_status": "MMRd", "mlh1_status": False, "sex": "male", "tissue_site": "right", "pathological_stage": 4, "tumor_grade": "high"},
        "C167":  {"MMR_status": "MMRd", "mlh1_status": True, "sex": "female", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "high"},
        "C168":  {"MMR_status": "MMRd", "mlh1_status": True, "sex": "male", "tissue_site": "right", "pathological_stage": 2, "tumor_grade": "low"},
        "C169":  {"MMR_status": "MMRd", "mlh1_status": False, "sex": "male", "tissue_site": "right", "pathological_stage": 2, "tumor_grade": "low"},
        "C170":  {"MMR_status": "MMRd", "mlh1_status": True, "sex": "female", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "low"},
        "C173":  {"MMR_status": "MMRd", "mlh1_status": True, "sex": "female", "tissue_site": "left", "pathological_stage": 4, "tumor_grade": "high"},
    
        "C103":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "male", "tissue_site": "left", "pathological_stage": 2, "tumor_grade": "low"},
        "C104":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "male", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "low"},
        "C105":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "male", "tissue_site": "right", "pathological_stage": 4, "tumor_grade": "low"},
        "C107":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "male", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "low"},
        "C112":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "male", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "low"},
        "C113":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "male", "tissue_site": "right", "pathological_stage": 2, "tumor_grade": "low"},
        "C124":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "female", "tissue_site": "left", "pathological_stage": 4, "tumor_grade": "high"},
        "C125":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "male", "tissue_site": "left", "pathological_stage": 3, "tumor_grade": "low"},
        "C126":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "female", "tissue_site": "right", "pathological_stage": 4, "tumor_grade": "low"},
        "C129":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "female", "tissue_site": "left", "pathological_stage": 3, "tumor_grade": "low"},
        "C133":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "female", "tissue_site": "right", "pathological_stage": 4, "tumor_grade": "low"},
        "C134":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "male", "tissue_site": "left", "pathological_stage": 4, "tumor_grade": "low"},
        "C135":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "male", "tissue_site": "right", "pathological_stage": 1, "tumor_grade": "low"},
        "C136":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "male", "tissue_site": "left", "pathological_stage": 3, "tumor_grade": "low"},
        "C140":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "male", "tissue_site": "left", "pathological_stage": 1, "tumor_grade": "low"},
        "C145":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "female", "tissue_site": "right", "pathological_stage": 4, "tumor_grade": "high"},
        "C149":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "female", "tissue_site": "right", "pathological_stage": 4, "tumor_grade": "low"},
        "C150":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "male", "tissue_site": "left", "pathological_stage": 3, "tumor_grade": "low"},
        "C153":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "female", "tissue_site": "left", "pathological_stage": 2, "tumor_grade": "low"},
        "C155":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "female", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "low"},
        "C157":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "female", "tissue_site": "left", "pathological_stage": 4, "tumor_grade": "low"},
        "C159":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "male", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "high"},
        "C160":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "female", "tissue_site": "right", "pathological_stage": 4, "tumor_grade": "high"},
        "C161":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "male", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "low"},
        "C162":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "male", "tissue_site": "right", "pathological_stage": 4, "tumor_grade": "low"},
        "C166":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "male", "tissue_site": "left", "pathological_stage": 3, "tumor_grade": "low"},
        "C171":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "male", "tissue_site": "right", "pathological_stage": 2, "tumor_grade": "low"},
        "C172":  {"MMR_status": "MMRp", "mlh1_status": False, "sex": "female", "tissue_site": "right", "pathological_stage": 3, "tumor_grade": "low"},
    }
    
    df_donor_meta = pd.DataFrame.from_dict(donor_metadata).T
    for column in df_donor_meta.columns:
        if column == "MMR_status":
            continue
        adata.obs[column] = adata.obs["donor_id"].map(df_donor_meta[column])

    # C132: the paper reports "female" while the raw TSV metadata say "male".
    # Keep the source discrepancy explicit instead of silently choosing one.
    adata.obs.loc[adata.obs["donor_id"].eq("C132"), "sex"] = "mismatch"

    return adata


def subset_to_epiT_and_unique_gene_id(adata):
    mask_epiT = (adata.obs["cell_type"] == "EpiT").values

    # we saw during the exploration that the duplicates with 0 counts always in 2nd place -> keep first
    mask_duplicated_genes = adata.var["gene_id"].duplicated().values

    X_filtered = adata.X[mask_epiT][:, ~mask_duplicated_genes].copy()
    obs_filtered = adata.obs.loc[mask_epiT].copy()
    var_filtered = adata.var.loc[~mask_duplicated_genes].copy()
    
    adata = ad.AnnData(X=X_filtered, obs=obs_filtered, var=var_filtered)
    adata.var.index = adata.var["gene_id"]
    adata.var.index.name = None

    return adata

def subset_to_gene_intersection(adata, gene_file_path):
    gene_df = pd.read_csv(gene_file_path)
    assert gene_df["gene_id"].isin(adata.var["gene_id"]).all()

    gene_mask = adata.var["gene_id"].isin(gene_df["gene_id"])
    adata = adata[:, gene_mask].copy()
    adata.var = adata.var.set_index("gene_symbol", drop=False)
    adata.var.index.name = None

    adata.obs["c_value"] = adata.obs["biosample_id"].str.extract(r"_(c\d+)_")

    return adata

def cols_to_cat(adata):
    cols = adata.obs.select_dtypes(include=["object"]).columns.tolist()
    cols.remove("cell_id")
    for var in cols:
        adata.obs[var] = adata.obs[var].astype("category")

    adata.obs["mlh1_status"] = adata.obs["mlh1_status"].astype(bool)
    
    return adata
