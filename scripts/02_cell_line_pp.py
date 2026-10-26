"""
Cell Line Preprocessing

Current approach:
- Load the csv data and save it in a better file format (h5ad could even be pickle) holding only the needed cells x genes
- Filter down to the gene intersection
- Filter down to CRC cell lines
- Filter down to cell lines with > 50 cells
- Attach msi status metadata and prepare batch_key batch
- save category cols as dtype category

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
from src import cell_line_preprocessing as pp

def main():
    config = load_config()

    cell_line_csv_path = config["paths"]["raw_data"]["cell_line"]
    gene_intersection_path = config["paths"]["processed"]["genes_combined"]
    final_save_path = config["paths"]["processed"]["pp_cell_line"]

    # Select CRC columns before constructing AnnData; the raw file contains
    # cells from all cancer types.
    adata = pp.load_cell_line_csv_as_adata(
        cell_line_csv_path, cell_line_contains="LARGE_INTESTINE"
    )
    adata = pp.subset_to_gene_intersection_with_gene_id(adata, gene_intersection_path)
    adata = pp.rename_cell_lines(adata)
    adata = pp.remove_cell_lines_lower_50_cells(adata)
    adata = pp.attach_extra_metadata(adata)
    adata = pp.cols_to_cat(adata)

    adata.write_h5ad(final_save_path)
    
if __name__ == "__main__":
    main()
