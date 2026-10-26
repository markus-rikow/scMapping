"""
Gene Selection Strategy

Current approach:
- Use intersection of genes across the datasets.

Rationale:
- Ensures consistent feature space across datasets.
- Reduces memory usage and avoids missing gene issues during integration.

Caveats / Future improvements:
- May discard informative dataset-specific genes.
- Consider expanding gene set (we have a version mismatch between the genes e.g. ACRC/GCNA) if integration performance is suboptimal.
    - Main issue is that cell line data is only gene symbols and tumor data as well as gene annotation file have different gene symbols (e.g. ACRC vs GCNA). I tried to resolve this by mapping the cell line gene symbols to the gene ids but as the ACRC gene symbol is not present in the gene annotation file, it was not possible to resolve this issue.

Revisit if:
- Cell line ↔ tumor mapping remains poor
- Important marker genes are missing after intersection
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config
from src import gene_intersection as gi

def main():
    config = load_config()

    tumor_path = config["paths"]["raw_data"]["tumor"]
    cell_line_path = config["paths"]["raw_data"]["cell_line"]
    gene_anno_path = config["paths"]["raw_data"]["gene_anno"]
    output_path = config["paths"]["processed"]["genes_combined"] 

    df_gene_anno = gi.load_gene_anno(gene_anno_path)
    df_cell_line = gi.load_and_prepare_cell_line_genes(cell_line_path, df_gene_anno)
    df_tumor = gi.load_and_prepare_tumor_genes(tumor_path)
    df_combined = gi.combine_genes(df_cell_line, df_tumor, df_gene_anno)

    df_combined.to_csv(output_path, index=False)
    
    
if __name__ == "__main__":
    main()