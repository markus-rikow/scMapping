"""Reusable plots for the biological-plausibility notebooks."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


def plot_leiden_sweep(adata, prefix, resolutions):
    """Plot UMAPs for a sweep of Leiden resolutions plus Pelka labels and cell lines."""
    cols = [f"{prefix}_leiden_res{res}" for res in resolutions]
    sc.pl.umap(adata, color=cols, legend_loc="on data")
    sc.pl.umap(adata, color="label")
    sc.pl.umap(
        adata,
        mask_obs=adata.obs["dataset"].eq("cell_line"),
        color=["cell_line"],
        na_in_legend=False,
        size=10,
        alpha=0.45,
    )


def _match_key(matches, cell_line):
    if cell_line in matches.index:
        return cell_line
    prefixed = f"line_{cell_line}"
    return prefixed if prefixed in matches.index else cell_line


def add_match_boxes(ax, matrix, matches):
    """Outline the primary and runner-up subpopulation for each cell line."""
    for y, cell_line in enumerate(matrix.index):
        key = _match_key(matches, cell_line)
        for group, color, linestyle in (
            (str(matches.loc[key, "primary_match"]), "black", "-"),
            (str(matches.loc[key, "second_match"]), "dimgray", "--"),
        ):
            if group in matrix.columns:
                x = matrix.columns.get_loc(group)
                ax.add_patch(
                    Rectangle(
                        (x, y),
                        1,
                        1,
                        fill=False,
                        edgecolor=color,
                        linewidth=1.8,
                        linestyle=linestyle,
                    )
                )


def plot_composition_and_margin(
    composition,
    matches,
    tumor_order,
    line_order,
    path=None,
    title="Tumor-only kNN composition",
):
    values = composition.loc[line_order, tumor_order]
    fig, (ax_heat, ax_margin) = plt.subplots(
        ncols=2,
        figsize=(18, max(5, 0.55 * len(line_order))),
        gridspec_kw={"width_ratios": [3.8, 1.4]},
    )
    sns.heatmap(
        values,
        ax=ax_heat,
        cmap="Blues",
        vmin=0,
        vmax=max(0.25, float(values.to_numpy().max())),
        annot=True,
        fmt=".2f",
        linewidths=0.3,
        cbar_kws={"label": "Mean tumor-neighbor fraction"},
    )
    add_match_boxes(ax_heat, values, matches)
    ax_heat.set_xlabel("Tumor subpopulation")
    ax_heat.set_ylabel("Cell line")
    ax_heat.set_title(title)
    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=90)

    margins = matches.loc[line_order, "margin"]
    colors = [
        "tab:red"
        if bool(matches.loc[name, "low_proximity"])
        else "tab:orange"
        if bool(matches.loc[name, "multiple_viable"])
        else "tab:blue"
        for name in line_order
    ]
    ax_margin.barh(np.arange(len(line_order)), margins, color=colors)
    ax_margin.invert_yaxis()
    ax_margin.axvline(0.10, color="black", linestyle="--", linewidth=1)
    ax_margin.set_yticks(np.arange(len(line_order)))
    ax_margin.set_yticklabels(line_order, fontsize=8)
    ax_margin.yaxis.tick_right()
    ax_margin.set_xlim(0, max(margins) * 1.18)
    for y, value in enumerate(margins):
        ax_margin.text(value + 0.005, y, f"{value:.2f}", va="center", fontsize=7)
    ax_margin.set_xlabel("Primary minus runner-up fraction")
    ax_margin.set_title("Match confidence")
    ax_margin.grid(axis="x", alpha=0.25)

    fig.legend(
        handles=[
            Line2D([0], [0], color="tab:blue", lw=6, label="unflagged"),
            Line2D([0], [0], color="tab:orange", lw=6, label="multiple viable"),
            Line2D([0], [0], color="tab:red", lw=6, label="low proximity"),
            Line2D([0], [0], color="black", lw=2, label="primary match"),
            Line2D([0], [0], color="dimgray", lw=2, linestyle="--", label="runner-up"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_distance_heatmap(
    distance_matrix,
    matches,
    path=None,
    title="Median latent distance from each cell line to each subpopulation",
):
    fig, ax = plt.subplots(
        figsize=(
            max(12, 0.65 * len(distance_matrix.columns)),
            max(5, 0.55 * len(distance_matrix)),
        )
    )
    sns.heatmap(
        distance_matrix,
        ax=ax,
        cmap="Blues_r",
        vmin=1.0,
        vmax=3.0,
        linewidths=0.3,
        cbar_kws={"label": "Median nearest-cell Euclidean distance", "extend": "max"},
    )
    add_match_boxes(ax, distance_matrix, matches)
    ax.set_xlabel("Tumor subpopulation")
    ax.set_ylabel("Cell line")
    ax.set_title(title)
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_line_umap(
    adata,
    line_order,
    path=None,
    basis="X_umap",
    title="Cell line cells in the latent space",
):
    umap = adata.obsm[basis]
    is_line = adata.obs["dataset"].eq("cell_line").to_numpy()
    is_tumor = ~is_line
    labels = adata.obs.loc[is_line, "cell_line"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        umap[is_tumor, 0],
        umap[is_tumor, 1],
        s=1.5,
        c="#d9d9d9",
        linewidths=0,
        rasterized=True,
    )
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#279e68",
        "#d62728",
        "#aa40fc",
        "#8c564b",
        "#e377c2",
        "#b5bd61",
        "#17becf",
        "#aec7e8",
    ]
    clean_order = [str(line).replace("line_", "") for line in line_order]
    for color, cell_line in zip(palette, clean_order):
        mask = labels.eq(cell_line).to_numpy()
        if mask.any():
            ax.scatter(
                umap[is_line][mask, 0],
                umap[is_line][mask, 1],
                s=10,
                c=[color],
                linewidths=0,
                label=cell_line,
            )
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=8,
        markerscale=1.5,
    )
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_subpopulation_umap(
    adata,
    group_col,
    label_col,
    path=None,
    basis="X_umap",
    subpop_title="Tumor subpopulations",
    group_order=None,
):
    """Plot tumor subpopulations and Pelka labels on a tumor-only coloured UMAP.

    ``group_order`` controls the categorical order and therefore the scanpy
    colour assignment. Pass the numeric tumor order (for example ``tumor_0``
    through ``tumor_17``) to keep the palette assignment stable across figures.
    """
    tumor_mask = adata.obs["dataset"].eq("tumor")
    group_values = adata.obs[group_col].astype("string").where(tumor_mask)
    if group_order is None:
        group_plot = group_values.astype("category")
    else:
        group_plot = pd.Categorical(group_values, categories=list(group_order))
    label_plot = (
        adata.obs[label_col]
        .astype("category")
        .where(tumor_mask)
        .cat.remove_unused_categories()
    )

    adata.obs["_bio_subpop_plot"] = group_plot
    adata.obs["_bio_label_plot"] = label_plot
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
        sc.pl.embedding(
            adata,
            basis=basis,
            color="_bio_subpop_plot",
            ax=axes[0],
            show=False,
            legend_loc="on data",
            legend_fontsize=8,
            frameon=False,
            mask_obs=tumor_mask,
            size=4,
            na_in_legend=False,
            title=subpop_title,
        )
        sc.pl.embedding(
            adata,
            basis=basis,
            color="_bio_label_plot",
            ax=axes[1],
            show=False,
            legend_loc="right margin",
            legend_fontsize=8,
            frameon=False,
            size=4,
            na_color="#d9d9d9",
            title="Pelka state labels",
            na_in_legend=False,
        )
        plt.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=300, bbox_inches="tight")
    finally:
        del adata.obs["_bio_subpop_plot"], adata.obs["_bio_label_plot"]
    return fig

def plot_tumor_gsea_heatmap(
    results_gsea_tumor,
    selected_subpops,
    n_top=20,
    path=None,
    title="Hallmark programs of candidate tumor subpopulations",
):
    focus = results_gsea_tumor.loc[
        results_gsea_tumor["subpopulation"].isin(selected_subpops)
    ].copy()
    pathway_strength = (
        focus.groupby("pathway")["nes"]
        .apply(lambda values: values.abs().max())
        .sort_values(ascending=False)
    )
    pathways = pathway_strength.head(n_top).index.tolist()
    heatmap = (
        focus[focus["pathway"].isin(pathways)]
        .pivot(index="subpopulation", columns="pathway", values="nes")
        .reindex(index=selected_subpops, columns=pathways)
    )

    fig, ax = plt.subplots(figsize=(18, 5.5))
    sns.heatmap(
        heatmap,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".1f",
        linewidths=0.3,
        cbar_kws={"label": "GSEA normalized enrichment score (NES)"},
    )
    ax.set_title(title)
    ax.set_xlabel("Hallmark pathway")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.tick_params(axis="y", labelrotation=0)
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_signature_match(
    signature_sets,
    matches,
    line_order,
    tumor_order,
    path=None,
):
    """Plot cell-line vs tumor GSEA signature correlation panels with match boxes and best-match stars."""
    row_order = [str(line).replace("line_", "") for line in line_order]
    fig, axes = plt.subplots(
        ncols=len(signature_sets),
        figsize=(9.5 * len(signature_sets), 5.8),
        squeeze=False,
    )
    for ax, (name, similarity) in zip(axes[0], signature_sets.items()):
        values = similarity.loc[row_order, tumor_order]
        image = ax.imshow(
            values.to_numpy(float),
            aspect="auto",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_xticks(np.arange(len(tumor_order)))
        ax.set_xticklabels(tumor_order, rotation=90, fontsize=8)
        ax.set_yticks(np.arange(len(row_order)))
        ax.set_yticklabels(row_order, fontsize=9)
        ax.set_xlabel("Tumor subpopulation")
        ax.set_title(f"Cell line vs tumor signature correlation: {name}", fontsize=10)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02).set_label("Spearman rho")

        for y, cell_line in enumerate(row_order):
            key = _match_key(matches, cell_line)
            for group, color, linestyle in (
                (str(matches.loc[key, "primary_match"]), "black", "-"),
                (str(matches.loc[key, "second_match"]), "dimgray", "--"),
            ):
                if group in tumor_order:
                    x = list(tumor_order).index(group)
                    ax.add_patch(
                        Rectangle(
                            (x - 0.5, y - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor=color,
                            linewidth=1.8,
                            linestyle=linestyle,
                        )
                    )
            best = values.loc[cell_line].idxmax()
            ax.plot(
                list(tumor_order).index(best),
                y,
                marker="*",
                color="white",
                markeredgecolor="black",
                markersize=11,
                zorder=5,
            )

        previous_primary = None
        for y, cell_line in enumerate(row_order):
            key = _match_key(matches, cell_line)
            primary = str(matches.loc[key, "primary_match"])
            if previous_primary is not None and primary != previous_primary:
                ax.axhline(y - 0.5, color="black", linewidth=1.2)
            previous_primary = primary

    axes[0, 0].set_ylabel("Cell line")
    fig.legend(
        handles=[
            Line2D([0], [0], color="black", lw=2, label="latent primary match"),
            Line2D(
                [0], [0], color="dimgray", lw=2, linestyle="--", label="latent runner-up"
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="white",
                markeredgecolor="black",
                linestyle="",
                markersize=11,
                label="best-matching subpopulation",
            ),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_domain_shift(
    domain_shift_df, path=None, title="Hallmark domain shift", n_extreme=4
):
    selected = list(domain_shift_df.head(n_extreme).index) + list(
        domain_shift_df.tail(n_extreme).index
    )
    selected = sorted(
        set(selected),
        key=lambda pathway: -abs(domain_shift_df.loc[pathway, "difference"]),
    )
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(selected) + 2))
    sns.heatmap(
        domain_shift_df.loc[selected, ["difference"]],
        ax=ax,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.3,
        cbar_kws={"label": "MLM score difference (cell line - tumor)"},
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_domain_shift_representations(
    shift_wide, domain_shift_by_rep, path=None, n_extreme=4
):
    selected = set()
    for table in domain_shift_by_rep.values():
        selected.update(table.head(n_extreme).index)
        selected.update(table.tail(n_extreme).index)
    selected = sorted(
        selected,
        key=lambda pathway: -abs(shift_wide.loc[pathway, "full_uncorrected"]),
    )
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(selected) + 2))
    sns.heatmap(
        shift_wide.loc[selected],
        ax=ax,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.3,
        cbar_kws={"label": "MLM score difference (cell line - tumor)"},
    )
    ax.set_title("Domain shift of Hallmark MLM scores across representations")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_concordance(
    matrix, matches, path=None, title="Cell line / tumor concordance"
):
    fig, ax = plt.subplots(figsize=(18, 6))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.3,
        cbar_kws={"label": "Spearman correlation"},
    )
    add_match_boxes(ax, matrix, matches)
    ax.set_xlabel("Tumor subpopulation")
    ax.set_ylabel("Cell line")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(title)
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_concordance_representations(concordance_by_rep, matches, path=None):
    labels = list(concordance_by_rep.keys())
    fig, axes = plt.subplots(ncols=len(labels), figsize=(7.5 * len(labels), 6))
    for ax, label in zip(axes, labels):
        matrix = concordance_by_rep[label]
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.3,
            cbar=False,
            xticklabels=True,
            yticklabels=(label == labels[0]),
        )
        add_match_boxes(ax, matrix, matches)
        ax.set_title(label)
        ax.set_xlabel("Tumor subpopulation")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        ax.set_ylabel("Cell line" if label == labels[0] else "")
    fig.subplots_adjust(right=0.9, wspace=0.25)
    colorbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(axes[0].collections[0], cax=colorbar_ax).set_label(
        "Spearman correlation of MLM pathway scores"
    )
    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_noprolif_comparison(
    all_pathway,
    no_prolif,
    matches,
    n_shared,
    n_noprolif,
    path=None,
):
    labels = list(all_pathway.keys())
    fig, axes = plt.subplots(2, len(labels), figsize=(7.5 * len(labels), 8.5))
    for column, label in enumerate(labels):
        for row, (matrix, title) in enumerate(
            [
                (all_pathway[label], f"{label}: all {n_shared} shared pathways"),
                (
                    no_prolif[label],
                    f"{label}: {n_noprolif} pathways without proliferation",
                ),
            ]
        ):
            ax = axes[row, column]
            sns.heatmap(
                matrix,
                ax=ax,
                cmap="RdBu_r",
                center=0,
                vmin=-1,
                vmax=1,
                linewidths=0.3,
                cbar=False,
                xticklabels=True,
                yticklabels=(column == 0),
            )
            add_match_boxes(ax, matrix, matches)
            ax.set_title(title, fontsize=10)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
            ax.set_xlabel("Tumor subpopulation" if row == 1 else "")
            ax.set_ylabel("Cell line" if column == 0 else "")
    fig.subplots_adjust(right=0.9, wspace=0.25, hspace=0.35)
    colorbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(axes[0, 0].collections[0], cax=colorbar_ax).set_label(
        "Spearman correlation of MLM pathway scores"
    )
    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_pathway_similarity(
    similarity_by_method,
    matches,
    line_order,
    tumor_order,
    methods=("z_dist | tumor_sig", "centered_spearman | tumor_sig"),
    path=None,
):
    """Plot tumor z-space and tumor-centered Spearman pathway similarity heatmaps."""
    row_order = [str(line).replace("line_", "") for line in line_order]
    plot_methods = [m for m in methods if m in similarity_by_method]
    if not plot_methods:
        plot_methods = list(similarity_by_method.keys())[:2]

    fig, axes = plt.subplots(
        ncols=len(plot_methods),
        figsize=(9.5 * len(plot_methods), 5.8),
        squeeze=False,
    )
    for ax, method in zip(axes[0], plot_methods):
        use_z = method.startswith("z_dist")
        similarity = similarity_by_method[method].reindex(
            index=row_order, columns=tumor_order
        )
        if use_z:
            image = ax.imshow(
                similarity.to_numpy(float),
                aspect="auto",
                cmap="viridis",
                interpolation="nearest",
            )
            title = "Similarity in tumor z-space"
            colorbar_label = "similarity (-Euclidean distance)"
        else:
            image = ax.imshow(
                similarity.to_numpy(float),
                aspect="auto",
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                interpolation="nearest",
            )
            title = "Spearman correlation of centered pathway profiles"
            colorbar_label = "Spearman rho"

        ax.set_xticks(np.arange(len(tumor_order)))
        ax.set_xticklabels(tumor_order, rotation=90, fontsize=8)
        ax.set_yticks(np.arange(len(row_order)))
        ax.set_yticklabels(row_order, fontsize=9)
        ax.set_xlabel("Tumor subpopulation")
        ax.set_title(title, fontsize=11)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02).set_label(colorbar_label)

        for y, cell_line in enumerate(row_order):
            key = _match_key(matches, cell_line)
            for group, color, linestyle in (
                (str(matches.loc[key, "primary_match"]), "black", "-"),
                (str(matches.loc[key, "second_match"]), "dimgray", "--"),
            ):
                if group in tumor_order:
                    x = list(tumor_order).index(group)
                    ax.add_patch(
                        Rectangle(
                            (x - 0.5, y - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor=color,
                            linewidth=1.8,
                            linestyle=linestyle,
                        )
                    )
            best = similarity.loc[cell_line].idxmax()
            ax.plot(
                list(tumor_order).index(best),
                y,
                marker="*",
                color="white",
                markeredgecolor="black",
                markersize=11,
                zorder=5,
            )

        previous_primary = None
        for y, cell_line in enumerate(row_order):
            key = _match_key(matches, cell_line)
            primary = str(matches.loc[key, "primary_match"])
            if previous_primary is not None and primary != previous_primary:
                ax.axhline(y - 0.5, color="black", linewidth=1.2)
            previous_primary = primary

    axes[0, 0].set_ylabel("Cell line")
    fig.legend(
        handles=[
            Line2D([0], [0], color="black", lw=2, label="latent primary match"),
            Line2D(
                [0], [0], color="dimgray", lw=2, linestyle="--", label="latent runner-up"
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="white",
                markeredgecolor="black",
                linestyle="",
                markersize=11,
                label="best-scoring subpopulation",
            ),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig
