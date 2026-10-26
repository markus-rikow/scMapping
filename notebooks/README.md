# Notebooks

The notebooks document preprocessing, exploratory analysis, post-HPO validation, and the biological plausibility analyses.
The authoritative preprocessing and training pipeline is in `scripts/`.
The notebooks are the interactive record and the downstream analyses.

## Run order

1. Data preparation (scripts, not notebooks):

   ```bash
   ./run_preprocessing.sh
   ./run_hpo.sh
   ./run_ranking.sh
   ```

2. Preprocessing notebooks (`1_preprocessing/`): optional interactive counterparts of the preprocessing scripts.
   They document the steps but do not save the processed artifacts.

3. EDA (`2_eda/`): requires `data/processed/preprocessed_tumor.h5ad` and `data/processed/preprocessed_cell_line.h5ad`.
   Saves figures to `figures/EDA/`.

4. Post-HPO validation (`3_hpo_validation/`): requires the completed HPO runs under `models/hpo/`.
   Artifacts go to `models/validation/`.

5. Biological plausibility (`4_biological_plausibility/`): run in numeric order.
   `1_clustering_resolution` creates the shared artifact; notebooks `2`-`6` read it and the cached DE/GSEA/MLM results.

6. Exploration (`exploration/`): historical checks and inspection plots.
   These are not required for the main results and some depend on untracked model checkpoints.

## Shared artifact

The plausibility notebooks read:

```text
adata_config10_joint/config10_joint_3k.h5ad
```

The file is produced by `1_clustering_resolution.ipynb` from the processed data and the saved config-10 joint model.
It is excluded from Git because of its size.

## Biological plausibility inputs

The notebooks in `4_biological_plausibility/` also need the cleaned Hallmark GMT:

```text
data/Gene_sets/h_cleaned.all.v2023.1.Hs.symbols.gmt
```

They also need the joint model plus its HVG list under `models/hpo/`.
Notebook `3` additionally needs the scArches query model from config 47.
Cached DE, GSEA, and MLM results are reused when present under `models/validation/`, otherwise the notebooks recompute them.
