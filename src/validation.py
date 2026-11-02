import json
from pathlib import Path

import pandas as pd

from src import line_ranking


def line_table(run_dir, k=100):
    cells = pd.read_csv(Path(run_dir) / "cell_line_mixing.csv")
    line = line_ranking.line_metrics_from_cells(
        cells,
        f"mixing_k{k}",
        f"entropy_k{k}",
        f"coverage_k{k}",
        f"line_purity_k{k}",
    )
    return line.set_index("cell_line")


def summary_row(run_dir, k=100):
    line = line_table(run_dir, k=k)
    summary = json.loads((Path(run_dir) / "eval_summary.json").read_text())
    return {
        "mean_line_score": line["line_score"].mean(),
        "mean_mixing": line["mixing"].mean(),
        "mean_specificity": line["specificity"].mean(),
        "mean_coverage": line["coverage"].mean(),
        "mean_line_purity": line["line_purity"].mean(),
        "scib_Total": summary["Total"],
        "scib_Batch_correction": summary["Batch correction"],
        "scib_Bio_conservation": summary["Bio conservation"],
        "n_epochs": summary["n_epochs"],
    }
