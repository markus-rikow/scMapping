import pandas as pd
import anndata as ad
import numpy as np

from scipy import sparse


def load_large_csv_in_sparse_chunks(
    file_path, meta_df, cell_ids=None, chunk_size=1000, skiprows=2
):
    """Load selected cell columns without materializing the full dense matrix.

    The SCP542 file is genes x cells. Chunks are therefore joined horizontally
    after transposition, producing a cells x genes CSR matrix.
    """
    metadata_cell_ids = meta_df.columns.astype(str).tolist()
    requested_cell_ids = metadata_cell_ids if cell_ids is None else [str(x) for x in cell_ids]

    header = pd.read_csv(file_path, sep="\t", skiprows=skiprows, nrows=0)
    expected_width = len(metadata_cell_ids) + 1
    if len(header.columns) != expected_width:
        raise ValueError(
            f"Count matrix has {len(header.columns) - 1} cell columns, "
            f"but metadata has {len(metadata_cell_ids)}"
        )

    metadata_cell_id_set = set(metadata_cell_ids)
    missing_cell_ids = sorted(set(requested_cell_ids) - metadata_cell_id_set)
    if missing_cell_ids:
        raise ValueError(
            f"Requested cell IDs are absent from the metadata: {missing_cell_ids[:5]}"
        )

    # The third line is a metadata row (Pool_ID), not the true cell header. Use
    # column positions and restore the cell IDs from the first-line metadata.
    requested_cell_set = set(requested_cell_ids)
    selected_cell_ids = [
        cell_id for cell_id in metadata_cell_ids if cell_id in requested_cell_set
    ]
    metadata_positions = {
        cell_id: position + 1 for position, cell_id in enumerate(metadata_cell_ids)
    }
    selected_positions = [metadata_positions[cell_id] for cell_id in selected_cell_ids]
    if not selected_cell_ids:
        raise ValueError("No requested cell IDs were found in the count matrix")

    chunks = []
    gene_symbols = []
    chunk_iter = pd.read_csv(
        file_path,
        sep="\t",
        index_col=0,
        skiprows=skiprows,
        usecols=[0] + selected_positions,
        chunksize=chunk_size,
    )

    print(
        f"    Loading {len(selected_cell_ids)} selected cell columns in sparse chunks..."
    )
    for i, chunk in enumerate(chunk_iter, start=1):
        chunk.index = chunk.index.astype(str)
        chunk.columns = selected_cell_ids
        gene_symbols.extend(chunk.index.tolist())
        counts_chunk = chunk.astype("uint32", copy=False)
        chunks.append(sparse.csr_matrix(counts_chunk.to_numpy(copy=False)).T)
        if i % 25 == 0:
            print(f"    Processed {i} gene chunks...")

    if not chunks:
        raise ValueError("The count matrix contains no gene rows")

    counts = sparse.hstack(chunks, format="csr", dtype=np.uint32)
    return counts, np.asarray(gene_symbols), selected_cell_ids


def load_cell_line_csv_as_adata(file_path, cell_line_contains=None):
    metadata = pd.read_csv(file_path, sep="\t", index_col=0, nrows=2)

    all_cell_ids = metadata.columns.astype(str).tolist()
    if cell_line_contains is not None:
        cell_lines = metadata.loc["Cell_line"].astype(str)
        selected_cell_ids = cell_lines.index[
            cell_lines.str.contains(cell_line_contains, regex=False, na=False)
        ].tolist()
    else:
        selected_cell_ids = all_cell_ids

    counts, gene_symbols, selected_cell_ids = load_large_csv_in_sparse_chunks(
        file_path,
        metadata,
        cell_ids=selected_cell_ids,
        chunk_size=500,
        skiprows=2,
    )

    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=selected_cell_ids),
        var=pd.DataFrame(index=gene_symbols),
    )

    adata.var["gene_symbol"] = adata.var_names
    adata.obs["cell_id"] = adata.obs_names
    selected_metadata = metadata.loc[:, selected_cell_ids]
    adata.obs["pool_id"] = selected_metadata.loc["Pool_ID"].to_numpy()
    adata.obs["cell_line"] = selected_metadata.loc["Cell_line"].to_numpy()
    adata.obs["chemistry"] = "v2"

    return adata

def subset_to_gene_intersection_with_gene_id(adata, gene_file_path):
    gene_df = pd.read_csv(gene_file_path)
    gene_df = gene_df.set_index("gene_symbol")
    
    print(f"    Sanity check: All intersection genes in cell line data? {gene_df.index.isin(adata.var_names).all()}")

    adata.var["gene_id"] = adata.var["gene_symbol"].map(gene_df["gene_id"])
    gene_mask = adata.var["gene_id"].notna()
    adata = adata[:, gene_mask].copy()
    
    return adata

def rename_cell_lines(adata):
    adata.obs["cell_line"] = adata.obs["cell_line"].str.split("_").str[0]
    return adata

def remove_cell_lines_lower_50_cells(adata):
    cell_line_counts = adata.obs["cell_line"].value_counts()
    valid_cell_lines = cell_line_counts[cell_line_counts >= 50].index
    adata = adata[adata.obs["cell_line"].isin(valid_cell_lines)].copy()

    return adata

def attach_extra_metadata(adata):
    msi_cellosaurus = {
        "SNUC4": True,
        "CL34": True, # both: Sanger is MSS; 3 Studies say MSI
        "LS180": True, # MSI-high
        "HT55": False,
        "LS1034": False,
        "RCM1": False,
        "CCK81": True, # MSI-high
        "OUMS23": False,
        "HCC56": False, # problematic cell line: Contaminated. Shown to be a HSC-57 derivative
        "NCIH747": False, # both: one study MSS; Sanger and one study say MSI-low
        "SNU81": False,
        "SW1116": False,
        "SW1463": False
    }

    adata.obs["mmr_status"] = adata.obs["cell_line"].map(msi_cellosaurus)
    adata.obs["mmr_status"] = adata.obs["mmr_status"].map({
        True: "MMRd",
        False: "MMRp"
    })

    adata.obs["batch"] = adata.obs["pool_id"].copy()
    adata.obs["dataset"] = "cell_line"
    adata.obs["label"] = "Unknown"
    adata.obs["donor_id"] = "cell_line"

    return adata

def cols_to_cat(adata):
    obs_cols_to_category = ["pool_id", "cell_line", "chemistry", "mmr_status", "batch", "label", "donor_id"]
    for var in obs_cols_to_category:
        adata.obs[var] = adata.obs[var].astype("category")

    return adata
