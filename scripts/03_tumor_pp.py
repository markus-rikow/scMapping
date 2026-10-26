"""
Tumor Preprocessing

Current approach:
- Load the h5 data and transform it into a adata holding only the needed cells x genes
- Attach all the needed metadata which includes: 
    - donor_id
    - mmr_status
    - label (cell states)
    - chemistry
    - batch_key
    - and much more
- Filter down to only epiT cells using unique gene_ids
- Filter down to the gene intersection
- save category cols as dtype category (and the bool col mlh1_staus as bool)

Caveats / Future improvements:
- See gene set improvements in scripts/01_gene_intersection.py

Revisit if:
- more genes are necessary. Currently only intersection between tumor and cell line. Should we ever need the cell line (or tumor) genes completely (without the intersection) we would need to create the anndata objects then 
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config
from src import tumor_preprocessing as pp

def main():
    config = load_config()

    tumor_h5_path = config["paths"]["raw_data"]["tumor"]
    tumor_metadata_path = config["paths"]["raw_data"]["tumor_metadata"]
    tumor_label_path = config["paths"]["raw_data"]["tumor_label"]
    gene_intersection_path = config["paths"]["processed"]["genes_combined"]
    final_save_path = config["paths"]["processed"]["pp_tumor"]

    adata = pp.load_tumor_h5_as_adata(tumor_h5_path)
    adata = pp.attach_metadata(adata, tumor_metadata_path)
    adata = pp.attach_cell_state_label(adata, tumor_label_path, batch_style=config["preprocessing"]["tumor_batch_style"])
    adata = pp.attach_paper_extra_metadata(adata)
    adata = pp.subset_to_epiT_and_unique_gene_id(adata)
    adata = pp.subset_to_gene_intersection(adata, gene_intersection_path)
    adata = pp.cols_to_cat(adata)

    adata.write_h5ad(final_save_path)
    
if __name__ == "__main__":
    main()