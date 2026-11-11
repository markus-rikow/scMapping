# scMapping

Code for matching colorectal cancer (CRC) cell lines to transcriptional subpopulations of primary tumors from single-cell RNA-seq.

Tumor and cell line cells are embedded in a shared latent space with scVI/scANVI.
Tumor cells are clustered with Leiden to define subpopulations.
Each cell line is matched by the tumor-subpopulation composition of its latent nearest neighbors.
Two integration strategies are compared: a joint scANVI approach and a reference-query scArches approach.
The repository also contains the biological plausibility analyses (scVI-based differential expression, Hallmark gene set enrichment, and pathway-activity scoring).

## Repository layout

```text
figures/                    generated figures (not tracked)
logs/                       training logs (not tracked)
data/                       raw and processed data (not tracked)
models/                     HPO runs, model checkpoints, caches (not tracked)
reproducibility/            frozen conda environments (GPU and CPU)
config/config.yaml          paths, preprocessing, HPO, and plausibility settings
src/                        preprocessing, HPO, and ranking code
src/bio/                    shared clustering, matching, GSEA, MLM, concordance, and plot helpers
scripts/                    pipeline scripts (01-05) and preprocessing validation
notebooks/                  preprocessing, EDA, validation, plausibility
run_preprocessing.sh        preprocessing wrapper
run_hpo.sh                  HPO wrapper
run_ranking.sh              post-HPO ranking wrapper
```

`data/`, `models/`, `figures/`, `logs/`, and the saved matching artifact `adata_config10_joint/` are excluded from Git.
Raw data and trained models are therefore not part of the repository.

## Environment

The analysis uses Python 3.11. Two conda specifications are provided:

```text
reproducibility/scvi-environment.yml       GPU environment (torch 2.6.0+cu124), env `scvi`
reproducibility/scvi-environment-cpu.yml   CPU environment (torch 2.6.0+cpu), env `scvi-cpu`
```

Use `scvi` for the reference GPU setup, and `scvi-cpu` when no CUDA GPU is available (for example on a WSL host).
Install only one of them:

```bash
# GPU machine
conda env create -f reproducibility/scvi-environment.yml
conda activate scvi

# CPU-only machine
conda env create -f reproducibility/scvi-environment-cpu.yml
conda activate scvi-cpu

python -m pip check
```

The GPU environment bundles the CUDA 12.4 runtime through the PyTorch wheels.
A system-wide CUDA toolkit is not required, but the host driver must support CUDA 12.4 (Linux driver >= 550.54.14).
A GPU is strongly recommended as the HPO sweep is slow on CPU.

On hosts with an older system `libstdc++` (missing `GLIBCXX_3.4.29`, e.g. WSL), `import torch` fails inside numpy unless the environment library directory is on `LD_LIBRARY_PATH`:

```bash
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib -n scvi-cpu
conda activate scvi-cpu   # reactivate to apply
```

## Data

Place the raw files under `data/raw/` using the names expected by `config/config.yaml`.
The repository does not redistribute the data.

| File | Source |
| --- | --- |
| `GSE178341_crc10x_full_c295v4_submit.h5` | Tumor atlas, Single Cell Portal [SCP1162](https://singlecell.broadinstitute.org/single_cell/study/SCP1162), GEO [GSE178341](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178341) (Pelka et al., 2021) |
| `metatable_v3_fix_v3.tsv` | Donor and sample metadata for GSE178341 |
| `crc10x_tSNE_cl_global.tsv` | Epithelial cell-state labels for GSE178341 |
| `SCP542_UMIcount_data.txt` | Cell line atlas, Single Cell Portal [SCP542](https://singlecell.broadinstitute.org/single_cell/study/SCP542), GEO [GSE157220](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE157220) (Kinker et al., 2020) |
| `geneAnno.csv` | Gene annotation table with Ensembl IDs and gene symbols |

The biological plausibility notebooks additionally need the MSigDB Hallmark gene sets.
Download the v2023.1 Hallmark symbols GMT into `data/Gene_sets/` and remove the `HALLMARK_` prefix, keeping the expected file name:

```bash
sed 's/HALLMARK_//g' \
  data/Gene_sets/h.all.v2023.1.Hs.symbols.gmt \
  > data/Gene_sets/h_cleaned.all.v2023.1.Hs.symbols.gmt
```

The preprocessing step writes the processed objects to `data/processed/`: `genes_combined.csv`, `preprocessed_cell_line.h5ad`, `preprocessed_tumor.h5ad`, and `preprocessing_report.json`.

## Workflow

All wrappers set `PYTHONPATH` to the repository root and can be run from any directory.
Use the environment's Python (`conda activate scvi` or `conda activate scvi-cpu`).

### 1. Preprocessing

```bash
./run_preprocessing.sh
```

This builds the shared gene table, preprocesses both datasets, and writes the validation report.

### 2. Hyperparameter optimization

```bash
./run_hpo.sh --dry-run   # print the run plan without training
./run_hpo.sh             # run the full sweep
```

The sweep trains 51 configurations per approach (17 configurations at each of 3,000, 5,000, and 10,000 HVGs), for 102 runs total.
Runs are resumable through `DONE` markers, so an interrupted sweep continues where it stopped.
Outputs:

```text
models/hpo/configs.json         sampled configurations
models/hpo/runs/                one directory per run
models/hpo/results.csv          one row per completed run
```

### 3. Ranking

```bash
./run_ranking.sh
```

The fixed-k kNN ranking (`k=100`) reads the stored per-cell mixing tables.
The dynamic-k pass recomputes each run's latent and scores `k = min(n_cells, k_max)` per cell line.
Outputs in `models/hpo/`:

```text
ranking_primary_100.csv         fixed-k primary ranking
ranking_primary_dynamic.csv     dynamic-k ranking
line_metrics.csv                per-run, per cell line detail
best_models_primary_100.csv     fixed-k rank-1 model per approach
best_models_primary_dynamic.csv dynamic-k rank-1 model per approach
```

The line score is the geometric mean of the mixing, tumor-label specificity, and purity computed for each cell line.
The run score averages the ten cell line scores with equal weight.

### 4. Biological plausibility

Run the notebooks in `notebooks/4_biological_plausibility/` in numeric order.
`1_clustering_resolution` creates the shared artifact. The remaining notebooks
consume it and write their caches under `models/validation/` and figures under
`figures/BIOVAL/`. The notebooks are the analysis record and call the shared
`src/bio/` helpers directly.

## Notebooks

Notebooks are organized by stage and are meant to be run in order within each
folder. Executed notebooks are committed with their outputs so the main tables and
figures can be inspected without rerunning the pipeline.

| Folder | Notebook | What it does |
| --- | --- | --- |
| `1_preprocessing` | `01_gene_intersection` | Notebook counterpart of the gene-intersection script |
| | `02_cell_line_pp` | Notebook counterpart of the cell line preprocessing script |
| | `03_tumor_pp` | Notebook counterpart of the tumor preprocessing script |
| `2_eda` | `EDA_combined` | Unintegrated combined UMAP before batch correction |
| | `EDA_tumor` | Tumor QC, composition, donor/chemistry structure, network mixing |
| | `EDA_cl` | Cell line QC, composition, and unintegrated UMAP |
| `3_hpo_validation` | `HPOVAL_scvi_base` | Unsupervised scVI baseline for both approach winners |
| | `HPOVAL_nb` | NB versus ZINB likelihood sensitivity |
| | `HPOVAL_seeds` | Seed stability of the joint winner |
| | `HPOVAL_dataset_batchkey` | Model batch key sensitivity (`dataset` versus `batch`) |
| `4_biological_plausibility` | `1_clustering_resolution` | Selects the tumor-only Leiden resolution and seed, saves the shared artifact |
| | `2_candidate_matching` | Tumor-only kNN candidate matching and match-confidence views |
| | `3_tumor_programs` | Subpopulation DE, Hallmark GSEA, and the `tumor_4`/`tumor_5` contrast |
| | `4_cell_line_gsea_mlm_concordance` | Direct DE/GSEA and signature view, MLM scoring, and pathway concordance |
| | `5_mlm_representation_sensitivity` | MLM representation sensitivity (corrected HVG, uncorrected HVG, full gene set) |
| | `6_scarches_plausibility` | Short scArches matching and MLM concordance check |
| | `7_mlm_no_prolif_pathways` | Concordance without the shared proliferation pathways |
| | `8_mlm_full_heatmap` | Joint-versus-scArches full MLM group concordance |
| `exploration` | various | Historical QC, HVG, batch-key, UMAP, and model comparisons (committed with outputs) |

See [`notebooks/README.md`](notebooks/README.md) for the detailed run order and shared artifact flow.
The preprocessing notebooks are interactive counterparts of the authoritative scripts.
The four `3_hpo_validation` notebooks read the completed HPO runs and write their artifacts under `models/validation/`.

The plausibility notebooks share the anndata object:

```text
notebooks/4_biological_plausibility/adata_config10_joint/config10_joint_3k.h5ad
```

`1_clustering_resolution` creates it from the processed data and the saved config-10 joint model with its latent representations.
`2_candidate_matching` through `8_mlm_full_heatmap` read it from the notebook's working directory.
The artifact is excluded from Git because of its size.
Notebook outputs are stored in the notebooks. Standalone figure files are written under `figures/` and are excluded from Git.

## Selected models and key results

The fixed-k ranking selected:

| Approach | Config | HVGs | Layers | Hidden | Latent | Dispersion | Mean line score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Joint scANVI | 10 | 3,000 | 3 | 256 | 10 | gene-batch | 0.679 |
| Reference-query scArches | 47 | 10,000 | 1 | 256 | 10 | gene-batch | 0.520 |

The joint representation was carried forward as the primary analysis. The scArches representation serves as a reference-query sensitivity view.

At tumor-only Leiden resolution 0.9 with seed 3, the tumor cells form 18 subpopulations.
The combined artifact contains 110,293 cells: 108,131 tumor epithelial cells and 2,162 cells from ten CRC cell lines.
Nine of the ten cell lines have the proliferation-dominated Leiden clusters `tumor_4` or `tumor_5` as their primary latent match.
Five of these candidates are supported by the pathway-level analyses, three remain unresolved because the two subpopulations are close, one is weak and heterogeneous (LS1034), and OUMS23 has no convincing match.

Post-HPO validation found that semi-supervision helps matching (scVI scores below scANVI), that NB and ZINB likelihoods give nearly identical joint results while scArches prefers ZINB, that seed variation changes the joint line score by less than 0.01, and that using `dataset` as the model batch key collapses mixing.

## Outputs and ignored paths

Generated data, models, caches, figures, and logs are excluded through `.gitignore`: `data/`, `models/`, `figures/`, `logs/`, `scvi_log/`, `archive/`, `adata_config10_joint/`, and `notebooks/counterfactual/`.
The repository therefore contains code, documentation, and executed notebooks with their outputs.

## Citation

If you use this code, please cite the two source datasets:

- Pelka et al. (2021), *Cell* — colorectal cancer tumor atlas (SCP1162/GSE178341).
- Kinker et al. (2020), *Cancer Cell* — cell line atlas (SCP542/GSE157220).

Please also cite scvi-tools, scIB, scIB-metrics, gseapy, and decoupler where their methods are used.
