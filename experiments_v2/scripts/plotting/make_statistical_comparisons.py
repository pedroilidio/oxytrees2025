import sys
import argparse
import warnings
from pathlib import Path
from itertools import product

import yaml
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import mlflow
import click

# HACK
sys.path.insert(0, str(Path(__file__).resolve().parent))
from critical_difference_diagrams import (
    plot_critical_difference_diagram,
    _find_maximal_cliques,
)

BASEDIR = Path(__file__).resolve().parents[2]


METRIC_FORMATTING = {
    "test_average_precision_micro": "Micro AP",
    "test_roc_auc_micro": "Micro AUROC",
    "test_matthews_corrcoef_micro": "Micro MCC",
    "test_f1_micro": "Micro F1",
    "test_neg_label_ranking_loss": "Ranking Error",
    "test_neg_hamming_loss_micro": "Micro Hamming Loss",
}


def combine_LT_TL(original_data):
    data = original_data.copy()
    data = data.set_index(["dataset", "fold", "estimator", "hue"])
    data.columns = data.columns.str.split("__", expand=True)

    if not {"LT", "TL"} < set(data.columns.get_level_values(0)):
        warnings.warn(
            "LT and TL metrics are not present in the data. Skipping LT+TL combination."
        )
        return original_data

    # Add LT+TL as new level
    lttl = pd.concat({"LT+TL": (data["LT"] + data["TL"]) / 2}, axis=1)

    data = pd.concat([data, lttl], axis=1)
    data.columns = ("__".join(c) for c in data.columns)

    return data.reset_index()


def set_axes_size(w, h, ax=None):
    """https://stackoverflow.com/a/44971177
    w, h: width, height in inches
    """
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def plot_insignificance_bars(*, positions, sig_matrix, ystart=None, ax=None, **kwargs):
    ax = ax or plt.gca()
    ylim = ax.get_ylim()
    ystart = ystart or ylim[1]
    crossbars = []
    crossbar_props = {"marker": ".", "color": "k"} | kwargs
    bar_margin = 0.1 * (ystart - ylim[0])

    positions = pd.Series(positions)  # Standardize if ranks is dict

    # True if pairwise comparison is NOT significant
    adj_matrix = pd.DataFrame(
        1 - sp.sign_array(sig_matrix),
        index=sig_matrix.index,
        columns=sig_matrix.columns,
        dtype=bool,
    )
    crossbar_sets = _find_maximal_cliques(adj_matrix)

    # Sort by lowest rank and filter single-valued sets
    crossbar_sets = sorted(
        (x for x in crossbar_sets if len(x) > 1), key=lambda x: positions[list(x)].min()
    )

    def bar_intersects(bar1, bar2):
        return not (
            positions[list(bar1)].max() < positions[list(bar2)].min()
            or positions[list(bar1)].min() > positions[list(bar2)].max()
        )

    # Create stacking of crossbars: for each level, try to fit the crossbar,
    # so that it does not intersect with any other in the level. If it does not
    # fit in any level, create a new level for it.
    crossbar_levels: list[list[set]] = []
    for bar in crossbar_sets:
        for level, bars_in_level in enumerate(crossbar_levels):
            if not any(bar_intersects(bar, bar_in_lvl) for bar_in_lvl in bars_in_level):
                ypos = ystart + bar_margin * (level + 1)
                bars_in_level.append(bar)
                break
        else:
            ypos = ystart + bar_margin * (len(crossbar_levels) + 1)
            crossbar_levels.append([bar])

        crossbars.append(
            ax.plot(
                # Adding a separate line between each pair enables showing a
                # marker over each elbow with crossbar_props={'marker': 'o'}.
                [positions[i] for i in bar],
                [ypos] * len(bar),
                **crossbar_props,
            )
        )

    return crossbars


def make_text_table(
    data,
    fold_col,
    estimator_col,
    metric,
    sig_matrix,
    positions,
    round_digits=2,
    highlight_best=True,
    higher_is_better=True,
):
    # True if pairwise comparison is NOT significant
    adj_matrix = pd.DataFrame(
        1 - sp.sign_array(sig_matrix),
        index=sig_matrix.index,
        columns=sig_matrix.columns,
        dtype=bool,
    )

    table = (
        data.set_index(fold_col).groupby(estimator_col)[metric].agg(["mean", "std"]).T
    )

    percentile_ranks = data.groupby(fold_col)[metric].rank(pct=True)
    is_victory = percentile_ranks == 1

    percentile_ranks_stats = (
        percentile_ranks.groupby(data[estimator_col]).agg(["mean", "std"]).T
    )
    is_victory_stats = (
        is_victory.groupby(data[estimator_col]).agg(["mean", "std"]).T
    )  # How many times was this estimator the best?

    text_table = {}
    for row, row_name in (
        (table, metric),
        (percentile_ranks_stats, metric + "_rank"),
        (is_victory_stats, metric + "_victories"),
    ):
        text_table[row_name] = (
            row.round(round_digits).astype(str).apply(lambda r: "{} ({})".format(*r))
        )
    text_table = pd.concat(text_table).reorder_levels([1, 0]).sort_index()

    if not highlight_best:
        return text_table

    crossbar_sets = _find_maximal_cliques(adj_matrix)

    # Get top-ranked set
    best_group = positions.idxmax() if higher_is_better else positions.idxmin()
    best_group = best_group[0]  # Select group from (group, hue_col)

    for crossbar in crossbar_sets:
        if best_group in crossbar:
            best_crossbar = list(crossbar)
            break
    else:
        raise RuntimeError

    # Highlight top-ranked set, if it is not the only set
    if len(best_crossbar) < len(adj_matrix):
        # HTML bold
        text_table.loc[(best_crossbar, slice(None))] = text_table[best_crossbar].apply(
            lambda s: f"<b>{s}</b>"
        )

    return text_table


def iter_posthoc_comparisons(
    data,
    *,
    y_cols,
    estimator_col,
    fold_col,
    p_adjust,
    hue_col=None,
):
    all_blocks = set(data[fold_col].unique())

    estimators_per_fold = data.groupby(fold_col)[estimator_col].count()
    folds_to_drop = estimators_per_fold[
        estimators_per_fold < estimators_per_fold.max()
    ].index
    if not folds_to_drop.empty:  # FIXME: explain
        warnings.warn(
            "The following groups have missing blocks and will be removed"
            f" from the comparison analysis:\n{folds_to_drop}"
        )
        data = data[~data[fold_col].isin(folds_to_drop)]

    missing_blocks = (
        data.groupby(estimator_col)[fold_col]
        .unique()
        .apply(lambda x: all_blocks - set(x))
    )
    missing_blocks = missing_blocks.loc[missing_blocks.apply(len) != 0]

    if not missing_blocks.empty:
        warnings.warn(
            "The following groups have missing blocks and will be removed"
            f" from the comparison analysis:\n{missing_blocks}"
        )
        data = data[~data[estimator_col].isin(missing_blocks.index)]

    groups = data[estimator_col].unique()
    n_groups = len(groups)

    indices = [fold_col, estimator_col]
    if hue_col is not None:
        indices.append(hue_col)

    for metric in y_cols:
        if n_groups <= 1:
            warnings.warn(
                f"Skipping {metric} because there are not enough groups "
                f"({n_groups}) to perform a test statistic."
            )
            continue
        pvalue_crosstable = sp.posthoc_nemenyi_friedman(
        #     # pvalue_crosstable = sp.posthoc_conover_friedman(
            data,
            melted=True,
            y_col=metric,
            block_col=fold_col,
            block_id_col=fold_col,
            group_col=estimator_col,
            # p_adjust=p_adjust,
        )
        # pvalue_crosstable = sp.posthoc_wilcoxon(
        #     data,
        #     val_col=metric,
        #     group_col=estimator_col,
        #     p_adjust=p_adjust,
        #     correction=True,
        #     zero_method="zsplit",
        #     sort=True,
        # )

        mean_ranks = (
            data.set_index(indices)[metric]
            .groupby(level=fold_col)
            .rank(pct=True)
            .groupby(
                level=estimator_col if hue_col is None else [estimator_col, hue_col]
            )
            .mean()
        )

        yield metric, pvalue_crosstable, mean_ranks


def make_visualizations(
    data,
    estimator_col,
    pvalue_crosstable,
    mean_ranks,
    outdir,
    metric,
    omnibus_pvalue,
    hue_col=None,
):
    # Define base paths
    sigmatrix_outpath = outdir / f"significance_matrices/{metric}"
    cdd_outpath = outdir / f"critical_difference_diagrams/{metric}"
    boxplot_outpath = outdir / f"boxplots/{metric}"

    # Create directories
    sigmatrix_outpath.parent.mkdir(exist_ok=True, parents=True)
    cdd_outpath.parent.mkdir(exist_ok=True, parents=True)
    boxplot_outpath.parent.mkdir(exist_ok=True, parents=True)

    pvalue_crosstable.to_csv(sigmatrix_outpath.with_suffix(".tsv"), sep="\t")

    n_groups = pvalue_crosstable.shape[0]
    formatted_metric_name = METRIC_FORMATTING.get(metric, metric)

    if data.dataset.iloc[0] == "all_datasets":
        formatted_metric_name = "Percentile ranks of " + formatted_metric_name
        data = data.copy()

    title = f"{formatted_metric_name}\np = {omnibus_pvalue:.2e}"

    plt.title(title, wrap=True)

    # # plt.figure(figsize=[(n_groups + 2) / 2.54] * 2)
    # plt.figure()

    # plt.title(title, wrap=True)

    # ax, cbar = sp.sign_plot(
    #     pvalue_crosstable,
    #     annot=sp.sign_table(pvalue_crosstable),
    #     fmt="s",
    #     square=True,
    # )
    # cbar.remove()
    # set_axes_size(*[(n_groups + 2) / 2.54] * 2)
    # plt.savefig(sigmatrix_outpath.with_suffix(".png"), bbox_inches="tight", dpi=300)
    # plt.savefig(
    #     sigmatrix_outpath.with_suffix(".pdf"),
    #     transparent=True,
    #     bbox_inches="tight",
    # )
    # plt.close()

    # plt.figure(figsize=(6, 0.5 * n_groups / 2.54 + 1))
    plt.figure()

    plot_critical_difference_diagram(
        mean_ranks.droplevel(hue_col),
        pvalue_crosstable,
        crossbar_props={"marker": "."},
    )
    plt.title(title, wrap=True)
    set_axes_size(6, 0.25 * n_groups / 2.54 + 1)
    plt.savefig(cdd_outpath.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.savefig(
        cdd_outpath.with_suffix(".pdf"),
        transparent=True,
        bbox_inches="tight",
    )
    plt.close()

    # order = (
    #     mean_ranks
    #     .sort_values()
    #     .sort_index(level=hue_col, sort_remaining=False)
    #     .index
    #     .get_level_values(0)
    # )
    means = data.groupby(estimator_col)[metric].mean().sort_values()
    order = means.index

    # plt.figure(figsize=(0.3 * n_groups + 1, 3))
    plt.figure()

    ax = sns.boxplot(
        data=data,
        x=estimator_col,
        y=metric,
        hue=hue_col,
        order=order,
        linecolor="k",
        linewidth=1.5,
        showfliers=False,
        legend=False,
        # showmeans=True,
        showmeans=False,
        meanprops={
            "marker": "d",
            "markerfacecolor": "C1",
            "markeredgecolor": "w",
            "markersize": 5,
            "zorder": 100,
        },
    )
    sns.stripplot(
        ax=ax,
        data=data,
        x=estimator_col,
        y=metric,
        hue=hue_col,
        order=order,
        palette=["k"] * mean_ranks.index.get_level_values(hue_col).nunique(),
        # color="black",
        marker="o",
        size=3,
        legend=False,
    )

    positions = {
        label.get_text(): tick
        for label, tick in zip(ax.get_xticklabels(), ax.get_xticks())
    }

    plot_insignificance_bars(
        positions=positions,
        sig_matrix=pvalue_crosstable,
    )

    plt.title(title, wrap=True)
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=45, ha="right", fontsize="large")
    ax.xaxis.set_tick_params(width=1.5)
    ax.yaxis.set_tick_params(width=1.5)

    # Add "(mean)" label to each x label
    # x_labels = [
    #     label.get_text() + f" ({means[label.get_text()]:.2g})"
    #     for label in ax.get_xticklabels()
    # ]
    # ax.set_xticklabels(x_labels)

    # plt.ylim(bottom=plt.ylim()[0] - 0.1 * (plt.ylim()[1] - plt.ylim()[0]))
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(1.5)

    fac = 10 ** (1 - int(np.floor(np.log10(means.abs().max()))))

    for xtick in ax.get_xticks():
        plt.annotate(
            # xtick,
            # plt.ylim()[0],
            # means.iloc[xtick],
            f"{means.iloc[xtick] * fac:.0f}",
            # (xtick, plt.ylim()[0]),
            (xtick, means.iloc[xtick]),
            backgroundcolor="white",
            size="small",
            horizontalalignment="center",
            verticalalignment="center",
            bbox=dict(facecolor="white", edgecolor="k", pad=2, linewidth=1.5),
            xycoords="data",
            # xytext=(xtick, plt.ylim()[0]),
            # annotation_clip=False,
        )

    if fac != 1:
        # Annotate fac on the bottom right:
        plt.annotate(
            # f"×{fac:.0e}",
            f"{1 / fac:.0e}"[1:],
            (0.95, 0.05),
            xycoords="axes fraction",
            backgroundcolor="white",
            size="small",
            horizontalalignment="center",
            verticalalignment="center",
            bbox=dict(facecolor="white", edgecolor="k", pad=2, linewidth=1.5),
        )

    # set_axes_size(0.3 * n_groups + 1, 3)
    set_axes_size(0.3 * (n_groups + 1), 2.7)
    plt.savefig(boxplot_outpath.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.savefig(
        boxplot_outpath.with_suffix(".pdf"),
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def friedman_melted(data, *, index, columns, values):
    # Expand ("unmelt") to 1 fold per column on level 2, metrics on level 1
    pivot = data.pivot(index=index, columns=columns, values=values)

    if pivot.shape[0] < 3:
        warnings.warn(
            f"Dataset {data.name} has only {pivot.shape[0]} estimators, "
            "which is not enough for a Friedman test."
        )
        result = pd.DataFrame(
            index=np.unique(pivot.columns.get_level_values(0)),
            columns=["statistic", "pvalue"],
            dtype=float,
        )
        result["statistic"] = np.nan
        result["pvalue"] = 1.0
        return result

    # Apply Friedman test for each result metric
    result = pivot.T.groupby(level=0).apply(
        lambda x: pd.Series(stats.friedmanchisquare(*(x.values.T))._asdict())
    )

    return result


def plot_comparison_matrix(comparison_data: pd.DataFrame):
    comparison_table = comparison_data.unstack()
    order = comparison_table.mean(1).sort_values(ascending=False).index

    comparison_table = comparison_table.loc[:, (slice(None), order)]
    comparison_table = comparison_table.loc[order]
    # comparison_table = comparison_table.loc[comparison_table.isna().sum(1).sort_values().index]
    # comparison_table = comparison_table.loc[:, comparison_table.isna().sum(0).sort_values().index]
    sns.heatmap(comparison_table.effect_size, annot=True)


def make_statistical_comparisons(
    *,
    data: pd.DataFrame,
    estimator_col: str = "estimator",
    dataset_col: str = "dataset",
    fold_col: str = "fold",
    hue_col: str | None = None,
    estimator_subset: list | None = None,
    dataset_subset: list | None = None,
    metric_subset: list | None = None,
    main_outdir: Path = Path("statistical_comparisons"),
    renaming: dict | None = None,
):
    data = data.copy()
    metric_names = set(data.columns) - {dataset_col, estimator_col, fold_col, hue_col}

    data = data.rename(
        columns={dataset_col: "dataset", estimator_col: "estimator", fold_col: "fold"}
    )
    if hue_col is None:
        data["hue"] = "no_hue"  # HACK
    else:
        data = data.rename(columns={hue_col: "hue"})

    if estimator_subset is not None:
        data = data[data.estimator.isin(estimator_subset)]
    if dataset_subset is not None:
        data = data[data.dataset.isin(dataset_subset)]
    if metric_subset is not None:
        data = data.loc[
            :,
            data.columns.isin(metric_subset + ["estimator", "dataset", "fold", "hue"]),
        ]
        metric_names &= set(data.columns)

    print(
        "Original selection:"
        + "\n  estimator_subset:\n  - "
        + "\n  - ".join(estimator_subset or [])
        + "\n  dataset_subset:\n  - "
        + "\n  - ".join(dataset_subset or [])
        + "\n  metric_subset:\n  - "
        + "\n  - ".join(metric_subset or [])
        + "\nSelected data:"
        + "\n  estimator_subset:\n  - "
        + "\n  - ".join(data.estimator.unique())
        + "\n  dataset_subset:\n  - "
        + "\n  - ".join(data.dataset.unique())
        + "\n  metric_subset:\n  - "
        + "\n  - ".join(data.columns[data.dtypes == float])
    )

    if data.empty:
        raise ValueError(
            "No data selected. Please review the filter parameters specified:"
            + "\n  estimator_subset:\n  - "
            + "\n  - ".join(estimator_subset or [])
            + "\n  dataset_subset:\n  - "
            + "\n  - ".join(dataset_subset or [])
            + "\n  metric_subset:\n  - "
            + "\n  - ".join(metric_subset or [])
        )

    # Drop duplicated runs (should be ordered by start time)
    dup = data.duplicated(["dataset", "fold", "estimator"], keep="last")
    if dup.any():
        warnings.warn(
            "The following runs were duplicated and will be removed from the"
            f" analysis:\n{data[dup]}"
        )
        data = data[~dup]

    allsets_data = data.pivot(index=["estimator", "hue"], columns=["dataset", "fold"])

    missing_metrics = allsets_data.isna().all(axis="index")

    if missing_metrics.any():
        print(
            "The following metrics were missing for all CV folds and"
            " will be removed from the analysis:"
            f"\n\n{allsets_data.loc[:, missing_metrics]}"
        )
        allsets_data = allsets_data.loc[:, ~missing_metrics]
        data = allsets_data.stack(["dataset", "fold"], future_stack=True).reset_index()

    missing_mask = allsets_data.isna().any(axis="index")

    if missing_mask.any():
        print(
            "The following runs were not present for all CV folds and"
            " will not be considered for rankings across all datasets:"
            f"\n\n{allsets_data.loc[:, missing_mask]}"
        )
        allsets_data = allsets_data.loc[:, ~missing_mask]

    allsets_data = allsets_data.stack(
        ["dataset", "fold"], future_stack=True
    ).reset_index()

    allsets_data = (
        allsets_data.set_index(["dataset", "fold", "estimator", "hue"])  # Keep columns
        .groupby(level=["dataset", "fold"])
        .rank(pct=True)  # Rank estimators per fold
        .mul(100)  # Convert to percentile ranks
        .groupby(level=["dataset", "estimator", "hue"])  # groupby()
        .mean()  # Average ranks across folds for each estimator
        .rename_axis(index=["fold", "estimator", "hue"])  # 'dataset' -> 'fold'
        .reset_index()
        .assign(dataset="all_datasets")
    )

    data = pd.concat([allsets_data, data], ignore_index=True, sort=False)

    # Average LT and TL metrics
    data = combine_LT_TL(data)

    # Rename data
    if renaming is not None:
        if "estimator" in renaming:
            data["estimator"] = data["estimator"].replace(renaming["estimator"])
        if "dataset" in renaming:
            data["dataset"] = data["dataset"].replace(renaming["dataset"])
        if "metric" in renaming:
            data = data.rename(columns=renaming["metric"])

    # Calculate omnibus Friedman statistics per dataset
    friedman_statistics = data.groupby("dataset").apply(
        friedman_melted,
        columns="fold",
        index="estimator",
        values=data.columns[data.dtypes == float],
    )
    friedman_statistics["corrected_p"] = multipletests(
        friedman_statistics.pvalue.values,
        # method="holm",
        method="fdr_bh",
    )[1]

    if data.isna().any().any():
        warnings.warn("NaNs found in the data. Skipping.")
        data = data.dropna()

    main_outdir.mkdir(exist_ok=True, parents=True)
    friedman_statistics.to_csv(main_outdir / "test_statistics.tsv", sep="\t")

    # Make parallel coordinates plot
    dataset_means = (
        data.loc[data.dataset != "all_datasets"]
        .drop(columns=["hue", "fold"])
        .groupby(["dataset", "estimator"])
        .mean()
    )

    for metric_name in tqdm(dataset_means.columns, desc="Parallel coordinates"):
        # outdir = main_outdir / "radar_plots"
        # outdir.mkdir(exist_ok=True, parents=True)
        # plot_radar(
        #     data=dataset_means,
        #     out=outdir / f"{metric_name}.png",
        #     metric=metric_name,
        # )

        outdir = main_outdir / "parallel_coordinates"
        outdir.mkdir(exist_ok=True, parents=True)
        plot_parallel_coordinates(
            data=dataset_means,
            out=outdir / f"{metric_name}.png",
            metric=metric_name,
        )

    table_lines = []
    grouped = data.groupby("dataset", sort=False)

    # Make visualizations of pairwise estimator comparisons.
    for dataset_name, dataset_group in tqdm(grouped, total=grouped.ngroups):

        # Existence is assured by make_visualizations()
        outdir = main_outdir / dataset_name

        for metric, pvalue_crosstable, mean_ranks in iter_posthoc_comparisons(
            dataset_group,
            y_cols=dataset_group.columns[dataset_group.dtypes == float],
            estimator_col="estimator",
            fold_col="fold",  # different from the above will all sets
            # p_adjust="holm",
            p_adjust="fdr_bh",
            hue_col="hue",
        ):
            # print(f"  ==> Processing {dataset_name=} {metric=}")
            if pvalue_crosstable.isna().any().any():
                warnings.warn(
                    f"NaNs found in the pvalue crosstable for {metric=} on"
                    f" {dataset_name=}. Skipping."
                )
                continue
            omnibus_pvalue = friedman_statistics.loc[dataset_name, metric].pvalue

            make_visualizations(
                data=dataset_group,
                metric=metric,
                pvalue_crosstable=pvalue_crosstable,
                mean_ranks=mean_ranks,
                estimator_col="estimator",
                outdir=outdir,
                omnibus_pvalue=omnibus_pvalue,
                hue_col="hue",
            )
            table_line = make_text_table(
                data=dataset_group,
                fold_col="fold",
                estimator_col="estimator",
                metric=metric,
                sig_matrix=pvalue_crosstable,
                positions=mean_ranks,
                round_digits=2,
                highlight_best=(omnibus_pvalue < 0.05),
                higher_is_better=not metric.endswith("time"),
            )

            # table_lines[(dataset_name, metric)] = table_line
            table_line = pd.concat({dataset_name: table_line}, names=["dataset"])
            table_lines.append(table_line)

    table = (
        pd.concat(table_lines)
        .rename_axis(["dataset", "estimator", "score"])
        .unstack(level=2)  # Set metrics as columns
    )
    table.to_csv(main_outdir / "comparison_table.csv")
    table.to_html(main_outdir / "comparison_table.html", escape=False)
    (
        table.apply(
            lambda x: x.str.replace(r"<b>(.*?)</b>", r"\\textbf{\1}", regex=True)
        ).to_latex(main_outdir / "comparison_table.tex")
    )


def plot_crosstab(data, out):
    print("Plotting crosstab...")
    crosstab = pd.crosstab(data.dataset, data.estimator).T
    plt.figure()
    sns.heatmap(
        crosstab,
        annot=True,
        cbar_kws=dict(label="Number of runs"),
        xticklabels=True,
        yticklabels=True,
    )
    set_axes_size(0.3 * crosstab.shape[1], 0.3 * crosstab.shape[0])
    out.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.clf()
    print(f"Saved crosstab to {out}")


def plot_parallel_coordinates(data, out, metric):
    plt.figure(figsize=(12, 4))
    sns.pointplot(
        data=data,
        x="dataset",
        y=metric,
        hue="estimator",
        markers="d",
        order=data.groupby("dataset")[metric].mean().sort_values().index,
    )
    plt.xticks(rotation=45)
    plt.xlabel("")
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.grid(axis="y")
    plt.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.clf()


def plot_radar(data, out, metric):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    data = data.reset_index()
    order = data.groupby("dataset")[metric].mean().sort_values().index
    mapping = {name: i for i, name in enumerate(order)}
    codes = data["dataset"].map(mapping)
    data["coord"] = 2 * np.pi / len(order) * codes
    unique_coords = 2 * np.pi / len(order) * np.arange(len(order))

    sns.lineplot(
        data=data,
        x="coord",
        y=metric,
        hue="estimator",
        markers="d",
        dashes=False,
        ax=ax,
    )
    ax.set_xticks(unique_coords, labels=order)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.grid(axis="y")
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.clf()


@click.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the configuration file.",
)
@click.option(
    "--results-table",
    "-r",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the results table.",
    multiple=True,
)
@click.option(
    "--out-crosstab",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to save a table counting runs per dataset and estimator.",
    default=BASEDIR / "results/run_counts.png",
)
@click.option(
    "--renaming",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the YAML file containing name substitutions.",
)
def main(config, results_table, out_crosstab, renaming):
    """Generate statistical comparisons between run results."""

    data = pd.concat(map(pd.read_csv, results_table))
    out_crosstab = out_crosstab.with_suffix(".png")  # Add suffix if not present
    plot_crosstab(data, out_crosstab)

    config = yaml.safe_load(config.read_text())
    renaming = renaming and yaml.safe_load(renaming.read_text())

    data = data.set_index("validation_setting")

    for config_object in config:
        if not config_object.get("active", True):
            continue

        # Convert nested list of estimators to a flat dictionary indicating the
        # hue (color) for each estimator
        estimator_hue = {}
        for hue, estimator_names in enumerate(config_object["estimator"]):
            for estimator_name in estimator_names:
                estimator_hue[estimator_name] = str(hue)

        estimator_subset = sum(config_object["estimator"], [])

        for validation_setting in config_object["validation_setting"]:
            outdir = Path(config_object["out"]) / validation_setting
            cv_data = data.loc[validation_setting]

            plot_crosstab(
                cv_data[
                    cv_data.estimator.isin(estimator_subset)
                    & cv_data.dataset.isin(config_object["dataset"])
                ],
                outdir / "run_counts.png",
            )
            cv_data["hue"] = cv_data["estimator"].map(estimator_hue)

            make_statistical_comparisons(
                data=cv_data,
                estimator_subset=estimator_subset,
                dataset_subset=config_object["dataset"],
                metric_subset=config_object["scoring"],
                renaming=renaming,
                main_outdir=outdir,
                hue_col="hue",
            )


if __name__ == "__main__":
    main()
