"""
Hyperparameter optimization driver for joint scANVI vs. scArches.

The full HPO design (search space, fixed training parameters, AnnData keys,
kNN sweep, HVG grid, sampling counts, seed, scIB subset size) lives in the
`hpo:` section of `config/config.yaml` and is the single authority. This
driver samples `n_configs_per_hvg` unique hyperparameter configs per HVG
level (default 17 x [3000, 5000, 10000] = 51 configs) and trains both
approaches with the same configs (51 joint scANVI + 51 scArches = 102
models).

Runs are interleaved per config: config i is trained and evaluated as joint
scANVI first, then as scArches, then config i+1. Saved and evaluated models
therefore exist at every point of a multi-day run.

Run layout under `paths.hpo.runs` (models/hpo/runs):
    <num_hvg>/config_<id>/joint/     -> joint_scanvi/ model dir + eval artifacts
    <num_hvg>/config_<id>/scarches/  -> reference/ + query/ + eval artifacts

Each run dir contains:
- config.json, DONE marker (a run is done only when the marker exists;
  failed runs have no DONE and are retried on the next execution)
- scib_results.csv, cell_line_mixing.csv, cell_line_mixing_summary.csv,
  eval_summary.json
(the combined latent is not stored; it is recomputed from the saved model
and the run's `hvg_genes.txt` when needed)
`paths.hpo.results` accumulates one summary row per (config, approach).

`configs.json` stores the sampled configs together with a design fingerprint;
it is only reused when the fingerprint still matches config.yaml. Regenerate
it with `--configs-only --force` after any design change (this invalidates the
recorded config_ids, so it refuses to run while run artifacts exist). Extend
it with `--extend [N]` to append N new configs per HVG level without reusing
already-sampled hyperparameter combinations.

Examples:
    python scripts/04_hpo.py --configs-only        # sample + save configs only
    python scripts/04_hpo.py --dry-run             # print the run plan
    python scripts/04_hpo.py --extend              # add one more round of configs
    python scripts/04_hpo.py --approach joint      # joint approach only
    python scripts/04_hpo.py --hvg 3000 --config-id 0   # single run
"""
import argparse
import itertools
import json
import sys
import time
import traceback
from pathlib import Path

import anndata as ad
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config
from src import hpo
from src import hpo_pp_utils as ppu

# Sentinel for `--extend` given without a value (argparse would otherwise
# collapse it to None, the same as "flag not passed").
EXTEND_DEFAULT = object()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach", choices=hpo.APPROACHES + ["both"], default="both")
    parser.add_argument(
        "--hvg", type=int, nargs="+", default=None,
        help="HVG grid to run (default: all from config.yaml)",
    )
    parser.add_argument(
        "--config-id", type=int, default=None,
        help="Run only this single config_id (debugging).",
    )
    parser.add_argument("--configs-only", action="store_true",
                        help="Only (re)sample and save the config table, then exit.")
    parser.add_argument(
        "--extend", nargs="?", const=EXTEND_DEFAULT, type=int, default=None,
        help="Append N additional configs per HVG level to the config table "
             "(default N = n_configs_per_hvg), without reusing already-sampled "
             "hyperparameter combinations, then exit.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the run plan without training.")
    parser.add_argument("--force", action="store_true",
                        help="Allow --configs-only to overwrite configs.json "
                             "even though run artifacts already exist.")
    return parser.parse_args()


def describe_config(cfg):
    return (
        f"id={cfg['config_id']:02d} hvg={cfg['num_hvg']} "
        f"n_layers={cfg['n_layers']} n_hidden={cfg['n_hidden']} "
        f"n_latent={cfg['n_latent']} dispersion={cfg['dispersion']} "
        f"encode_covariates={cfg['encode_covariates']} "
        f"n_samples_per_label={cfg['n_samples_per_label']}"
    )


def build_run_plan(hvg_grid, configs, approaches, only_config_id=None):
    plan = []
    for cfg in configs:
        if only_config_id is not None and cfg["config_id"] != only_config_id:
            continue
        if only_config_id is not None:
            plan.append((cfg, approaches))
            continue
        if cfg["num_hvg"] in hvg_grid:
            plan.append((cfg, approaches))
    return plan


def sampling_meta(hpo_cfg):
    """Design fingerprint stored alongside configs.json; configs are only
    reused when it still matches the current config.yaml design."""
    return {
        "seed": int(hpo_cfg["seed"]),
        "hvg_grid": [int(h) for h in hpo_cfg["hvg_grid"]],
        "n_configs_per_hvg": int(hpo_cfg["n_configs_per_hvg"]),
        "search_space": hpo_cfg["search_space"],
        "fixed": hpo_cfg["fixed"],
    }


def save_configs(configs_path, hpo_cfg, configs):
    configs_path = Path(configs_path)
    configs_path.parent.mkdir(parents=True, exist_ok=True)
    configs_path.write_text(
        json.dumps({"meta": sampling_meta(hpo_cfg), "configs": configs}, indent=2)
    )


def load_configs(configs_path, hpo_cfg):
    """Load configs.json, verifying it was sampled under the current design.

    Returns None when the file does not exist, raises on format mismatch or a
    design change since sampling.
    """
    configs_path = Path(configs_path)
    if not configs_path.exists():
        return None
    data = json.loads(configs_path.read_text())
    if not isinstance(data, dict) or "configs" not in data or "meta" not in data:
        raise ValueError(
            f"{configs_path} uses the old flat config format. Regenerate it "
            "with `python scripts/04_hpo.py --configs-only --force`."
        )
    if data["meta"] != sampling_meta(hpo_cfg):
        raise ValueError(
            f"{configs_path} was sampled under a different design (seed, "
            "hvg_grid, n_configs_per_hvg, search_space or fixed differ from "
            "config.yaml). Regenerate it with "
            "`python scripts/04_hpo.py --configs-only --force`."
        )
    return data["configs"]


def load_or_sample_configs(configs_path, hpo_cfg):
    configs = load_configs(configs_path, hpo_cfg)
    if configs is not None:
        return configs
    configs = hpo.sample_configs(hpo_cfg, None, None, None)
    save_configs(configs_path, hpo_cfg, configs)
    return configs


def extend_configs(hpo_cfg, configs, n_configs_per_hvg_new):
    """Append new configs without reusing any already-sampled combination.

    `grid_index` tracks the cross-product slot, so draws stay without
    replacement across the whole run history. New configs continue the global
    `config_id` sequence and HVG levels keep the round-robin block order of
    the initial sampling.
    """
    n_per_hvg_new = int(n_configs_per_hvg_new)
    if n_per_hvg_new <= 0:
        raise ValueError("n_configs_per_hvg_new must be positive.")

    search_space = hpo_cfg["search_space"]
    fixed = hpo_cfg["fixed"]
    hvg_grid = list(hpo_cfg["hvg_grid"])
    seed = int(hpo_cfg["seed"])
    n_per_hvg = int(hpo_cfg["n_configs_per_hvg"])

    keys = list(search_space.keys())
    grid = list(itertools.product(*[search_space[k] for k in keys]))
    n_total = len(grid)

    used = {int(c["grid_index"]) for c in configs}
    remaining = sorted(set(range(n_total)) - used)
    n_new = n_per_hvg_new * len(hvg_grid)
    if n_new > len(remaining):
        raise ValueError(
            f"Cannot extend: only {len(remaining)} unused hyperparameter "
            f"combinations remain but {n_new} are requested."
        )

    start_id = max(int(c["config_id"]) for c in configs) + 1
    start_block = len(configs) // n_per_hvg

    rng = np.random.RandomState(seed)
    chosen = rng.choice(remaining, size=n_new, replace=False)

    extended = [dict(c) for c in configs]
    for j, gi in enumerate(chosen):
        block = start_block + j // n_per_hvg_new
        num_hvg = hvg_grid[block % len(hvg_grid)]
        cfg = {k: v for k, v in zip(keys, grid[int(gi)])}
        cfg.update(dict(fixed))
        cfg["config_id"] = start_id + j
        cfg["num_hvg"] = num_hvg
        cfg["seed"] = seed
        cfg["grid_index"] = int(gi)
        extended.append(cfg)
    return extended


def hpo_outputs_exist(runs_dir, results_path):
    """True when previous run artifacts could be invalidated by regenerating
    the config table (config_ids are global and would point to new configs)."""
    runs_dir = Path(runs_dir)
    results_path = Path(results_path)
    runs_exist = runs_dir.exists() and any(runs_dir.iterdir())
    results_exist = results_path.exists() and results_path.stat().st_size > 0
    return runs_exist or results_exist


def main():
    import logging
    import warnings

    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning_fabric").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
    logging.getLogger("lightning.fabric").setLevel(logging.WARNING)
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=r"The .* does not have many workers.*",
        category=UserWarning,
    )

    args = parse_args()
    cfg = load_config()
    hpo_cfg = cfg["hpo"]
    paths = cfg["paths"]

    seed = hpo_cfg["seed"]
    hvg_grid = args.hvg if args.hvg else hpo_cfg["hvg_grid"]
    approaches = hpo.APPROACHES if args.approach == "both" else [args.approach]

    torch.set_float32_matmul_precision("high")
    hpo.set_seed(seed)

    configs_path = Path(paths["hpo"]["configs"])
    runs_dir = Path(paths["hpo"]["runs"])
    results_path = Path(paths["hpo"]["results"])

    if args.configs_only:
        if args.extend is not None:
            sys.exit("--extend cannot be combined with --configs-only.")
        if args.hvg:
            sys.exit(
                "--hvg is a run filter; --configs-only always samples over the "
                "full hvg_grid from config.yaml. Edit config.yaml to change the grid."
            )
        if not args.force and hpo_outputs_exist(runs_dir, results_path):
            sys.exit(
                "Refusing to regenerate configs: run artifacts already exist under "
                "models/hpo/ and recorded config_ids would be invalidated. "
                "Pass --force to overwrite anyway."
            )
        configs = hpo.sample_configs(hpo_cfg, None, None, None)
        save_configs(configs_path, hpo_cfg, configs)
        print(f"Saved {len(configs)} configs to {configs_path}")
        for cfg in configs:
            print(f"  {describe_config(cfg)}")
        return

    if args.extend is not None:
        n_new = (
            hpo_cfg["n_configs_per_hvg"]
            if args.extend is EXTEND_DEFAULT
            else args.extend
        )
        configs = load_configs(configs_path, hpo_cfg)
        if configs is None:
            sys.exit(
                f"No config table found at {configs_path}; "
                "run `python scripts/04_hpo.py --configs-only` first."
            )
        extended = extend_configs(hpo_cfg, configs, n_new)
        added = extended[len(configs):]
        if args.dry_run:
            print(f"Would add {len(added)} configs "
                  f"({n_new} per HVG level, round-robin over {hpo_cfg['hvg_grid']}).")
            for cfg in added:
                print(f"  {describe_config(cfg)}")
            return
        save_configs(configs_path, hpo_cfg, extended)
        print(f"Extended config table: {len(configs)} -> {len(extended)} configs "
              f"({len(added)} added).")
        for cfg in added:
            print(f"  {describe_config(cfg)}")
        return

    configs = load_or_sample_configs(configs_path, hpo_cfg)

    plan = build_run_plan(hvg_grid, configs, approaches, args.config_id)
    n_runs = len(plan) * len(approaches)
    print(f"Plan: {n_runs} (config, approach) runs "
          f"({len(set(c['config_id'] for c, _ in plan))} unique configs).")

    if args.dry_run:
        for cfg, _ in plan:
            print(f"  {describe_config(cfg)} -> {approaches}")
        return

    # Load raw preprocessed data once (raw counts in X).
    adata_tumor = ad.read_h5ad(paths["processed"]["pp_tumor"])
    adata_cl = ad.read_h5ad(paths["processed"]["pp_cell_line"])
    print(f"Loaded tumor {adata_tumor.shape} / cell lines {adata_cl.shape}")

    scib_n_cells = hpo_cfg["scib_n_cells"]

    by_hvg = {}
    for cfg, _ in plan:
        by_hvg.setdefault(cfg["num_hvg"], []).append(cfg)
    by_hvg = {h: sorted(c, key=lambda c: c["config_id"]) for h, c in by_hvg.items()}

    for num_hvg, hvg_configs in by_hvg.items():
        hvg_genes = ppu.compute_combined_hvg(
            adata_tumor, adata_cl, num_hvg,
            flavor=hpo_cfg["hvg_flavor"],
            batch_key=hpo_cfg["hvg_batch_key"],
        )
        print(f"HVG {num_hvg}: {len(hvg_genes)} genes selected")
        hvg_dir = runs_dir / f"hvg{num_hvg}"
        hvg_dir.mkdir(parents=True, exist_ok=True)
        (hvg_dir / "hvg_genes.txt").write_text("\n".join(hvg_genes))

        tumor_hvg = adata_tumor[:, hvg_genes].copy()
        cl_hvg = adata_cl[:, hvg_genes].copy()
        tumor_hvg.layers["counts"] = tumor_hvg.X.copy()
        cl_hvg.layers["counts"] = cl_hvg.X.copy()

        combined = ppu.build_combined_adata(tumor_hvg, cl_hvg)
        print(f"Combined adata: {combined.shape}, "
              f"{combined.obs['dataset'].value_counts().to_dict()}")

        # Config-major, approach-minor: joint first, then scArches, so every
        # finished config leaves a saved and evaluated model behind.
        for cfg in hvg_configs:
            for approach in approaches:
                run_dir = runs_dir / f"hvg{num_hvg}" / f"config_{cfg['config_id']:02d}" / approach
                if hpo.is_run_done(run_dir):
                    print(f"  [skip] id={cfg['config_id']} {approach}")
                    continue

                print(f"  [train] {describe_config(cfg)} {approach}")
                t0 = time.time()
                try:
                    if approach == "joint":
                        model = hpo.train_joint_scanvi(combined, cfg, run_dir)
                    else:
                        _, model = hpo.train_scarches(
                            tumor_hvg, cl_hvg, cfg, run_dir
                        )
                    summary = hpo.evaluate(
                        model, combined, cfg, run_dir, approach,
                        scib_n_cells=scib_n_cells,
                        train_time_s=time.time() - t0,
                    )
                except Exception as exc:  # noqa: BLE001 - keep the sweep alive
                    print(f"  [error] id={cfg['config_id']} {approach}: {exc}")
                    traceback.print_exc()
                    hpo.mark_run_error(run_dir, str(exc))
                    continue

                hpo.mark_run_done(run_dir)
                hpo.append_results_row(results_path, summary)
                k_show = cfg["k_values"][0]
                print(f"    done in {summary['train_time_s']:.1f}s -> "
                      f"elbo={summary['elbo_validation']} "
                      f"epochs={summary['n_epochs']} "
                      f"best_epoch={summary['best_epoch']} "
                      f"mixing_{k_show}={summary[f'mixing_{k_show}']:.3f} "
                      f"coverage_{k_show}={summary[f'coverage_{k_show}']:.3f} "
                      f"entropy_{k_show}={summary[f'entropy_{k_show}']:.3f} "
                      f"line_purity_{k_show}={summary[f'line_purity_{k_show}']:.3f}")

    print(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
