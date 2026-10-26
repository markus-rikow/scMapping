import pandas as pd
import os
import h5py

def load_gene_anno(gene_anno_file):
    df_gene_anno = pd.read_csv(gene_anno_file, usecols=["ensembl_gene_id", "external_gene_name"])
    df_gene_anno["ensembl_gene_id_base"] = df_gene_anno["ensembl_gene_id"].str.split(".").str[0]
    return df_gene_anno

def load_and_prepare_cell_line_genes(cell_line_file, df_gene_anno):
    df_cell_line = pd.read_csv(cell_line_file, sep="\t", usecols=[0], index_col=0, skiprows=2)
    df_cell_line.index.name = "gene_symbol"

    symbol_to_ensembl = df_gene_anno.set_index("external_gene_name")["ensembl_gene_id_base"].to_dict()
    df_cell_line["gene_id"] = df_cell_line.index.map(
        lambda x: symbol_to_ensembl.get(x)
    )
    df_cell_line.dropna(inplace=True)

    return df_cell_line

def load_and_prepare_tumor_genes(tumor_file):
    data = h5py.File(tumor_file, "r")
    features = data["matrix"]["features"]
    gene_ids = [x.decode() if isinstance(x, bytes) else x for x in features["id"][:]]
    gene_names = [x.decode() if isinstance(x, bytes) else x for x in features["name"][:]]

    df_tumor = pd.DataFrame(gene_ids, columns=["gene_id"])
    df_tumor.index = gene_names
    df_tumor.index.name = "gene_symbol"
    df_tumor = df_tumor[df_tumor["gene_id"].str.startswith(("ENSG"))].copy()
    df_tumor["gene_id_base"] = df_tumor["gene_id"].str.split(".").str[0]

    # remove duplicates based on gene_id_base, keeping the first occurrence
    df_tumor = df_tumor[~df_tumor["gene_id_base"].duplicated(keep="first")].copy()
    df_tumor["gene_id"] = df_tumor["gene_id_base"].copy()
    df_tumor.drop(columns=["gene_id_base"], inplace=True)

    return df_tumor

def combine_genes(df_cell_line, df_tumor, df_gene_anno):
    df_combined = df_cell_line.merge(
        df_tumor,
        on="gene_id",
        how="inner"
    )

    ensembl_to_symbol = df_gene_anno.set_index("ensembl_gene_id_base")["external_gene_name"].to_dict()
    df_combined["gene_symbol"] = df_combined["gene_id"].map(
        lambda x: ensembl_to_symbol.get(x)
    )

    return df_combined