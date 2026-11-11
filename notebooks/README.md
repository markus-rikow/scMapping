# Notebooks

The notebooks document preprocessing, exploratory analysis, post-HPO validation,
and the biological plausibility analyses.
The authoritative preprocessing and training pipeline is in `scripts/`.
The notebooks are the interactive record and the downstream analyses.
Executed notebooks are committed with their outputs.

## Run order

1. Data preparation (scripts, not notebooks):

   ```bash
   ./run_preprocessing.sh
   ./run_hpo.sh
   ./run_ranking.sh
   ```

2. Preprocessing notebooks (`1_preprocessing/`): notebook counterparts of the
   preprocessing scripts.
   They call the same `src` functions and stop before writing the processed
   artifacts, so they can be inspected without changing the pipeline outputs.

3. EDA (`2_eda/`): requires `data/processed/preprocessed_tumor.h5ad` and
   `data/processed/preprocessed_cell_line.h5ad`.
   Saves figures to `figures/EDA/`.

4. Post-HPO validation (`3_hpo_validation/`): requires the completed HPO runs
   under `models/hpo/`.
   Artifacts go to `models/validation/`.

5. Biological plausibility (`4_biological_plausibility/`): run in numeric order.

   | Notebook | Role |
   | --- | --- |
   | `1_clustering_resolution` | Selects the tumor-only resolution and seed and writes the shared artifact |
   | `2_candidate_matching` | Tumor-only kNN composition, candidates, margins, k sensitivity |
   | `3_tumor_programs` | Subpopulation DE and Hallmark GSEA |
   | `4_cell_line_gsea_mlm_concordance` | Direct DE/GSEA and signature view, MLM scoring, pathway concordance |
   | `5_mlm_representation_sensitivity` | MLM representation sensitivity |
   | `6_scarches_plausibility` | Short scArches matching and MLM concordance check |
   | `7_mlm_no_prolif_pathways` | Proliferation-excluded concordance |
   | `8_mlm_full_heatmap` | Joint-versus-scArches full MLM group concordance |

   `1_clustering_resolution` creates the shared artifact. Notebooks `2`-`8`
   read it and the cached DE/GSEA/MLM results.

6. Exploration (`exploration/`): historical checks and inspection plots.
   These are not required for the main results and are committed with their
   outputs. Some depend on untracked model checkpoints.

## Shared artifact

The plausibility notebooks read:

```text
adata_config10_joint/config10_joint_3k.h5ad
```

The file is produced by `1_clustering_resolution.ipynb` from the processed data
and the saved config-10 joint model.
It is excluded from Git because of its size.

## Biological plausibility inputs

The notebooks in `4_biological_plausibility/` also need the cleaned Hallmark GMT:

```text
data/Gene_sets/h_cleaned.all.v2023.1.Hs.symbols.gmt
```

They also need the joint model plus its HVG list under `models/hpo/`.
Notebook `6_scarches_plausibility` additionally needs the scArches query model
selected by the fixed-k ranking.
Cached DE, GSEA, and MLM results are reused when present under
`models/validation/`, otherwise the notebooks recompute them.

## scArches check

The scArches notebook uses the same matching parameters and cluster-size screens
as the joint analysis (`k=50`, tumor-only Leiden resolution 0.9, and Leiden seed
0 selected by the `< 50` cell fragment and `<= 15%` max-share screens on the
scArches graph). It is a sensitivity check on the query representation, not a
second full biological analysis.
