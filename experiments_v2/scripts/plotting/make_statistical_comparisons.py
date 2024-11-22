import sys
import argparse
import warnings
from pathlib import Path

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


METRIC_FORMATTING = {
    "test_average_precision_micro": "Micro AP",
    "test_roc_auc_micro": "Micro AUROC",
    "test_matthews_corrcoef_micro": "Micro MCC",
    "test_f1_micro": "Micro F1",
    "test_neg_label_ranking_loss": "Ranking Error",
    "test_neg_hamming_loss_micro": "Micro Hamming Loss",
}


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
        # pvalue_crosstable = sp.posthoc_nemenyi_friedman(
        #     # pvalue_crosstable = sp.posthoc_conover_friedman(
        #     data,
        #     melted=True,
        #     y_col=metric,
        #     estimator_col=estimator_col,
        #     fold_col=fold_col,
        #     # p_adjust=p_adjust,
        # )
        pvalue_crosstable = sp.posthoc_wilcoxon(
            data,
            val_col=metric,
            group_col=estimator_col,
            p_adjust=p_adjust,
            correction=True,
            zero_method="zsplit",
            sort=True,
        )

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

    # HACK
    estimator_name = data[estimator_col].iloc[0].split("__")
    if len(estimator_name) > 1:
        drop = estimator_name[1]
        data = data.copy()
        pvalue_crosstable = pvalue_crosstable.copy()
        mean_ranks = mean_ranks.copy()
        data.loc[:, estimator_col] = data[estimator_col].str.split("__").str[0]
        pvalue_crosstable.index = pvalue_crosstable.index.str.split("__").str[0]
        pvalue_crosstable.columns = pvalue_crosstable.columns.str.split("__").str[0]
        mean_ranks.index = mean_ranks.index.set_levels(
            mean_ranks.index.get_level_values(estimator_col).str.split("__").str[0],
            level=estimator_col,
        )
    else:
        drop = None

    if data.dataset.iloc[0] == "all_datasets":
        # formatted_metric_name += "\n(mean percentile ranks)"
        formatted_metric_name += " (ranks)"
        data = data.copy()
        data[metric] *= 100

    title = f"{formatted_metric_name}\np = {omnibus_pvalue:.2e}"

    if drop is not None:
        title += f" | ILR = {drop}%"

    plt.title(title, wrap=True)

    # plt.figure(figsize=[(n_groups + 2) / 2.54] * 2)
    plt.figure()

    plt.title(title, wrap=True)

    ax, cbar = sp.sign_plot(
        pvalue_crosstable,
        annot=sp.sign_table(pvalue_crosstable),
        fmt="s",
        square=True,
    )
    cbar.remove()
    set_axes_size(*[(n_groups + 2) / 2.54] * 2)
    plt.savefig(sigmatrix_outpath.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.savefig(
        sigmatrix_outpath.with_suffix(".pdf"),
        transparent=True,
        bbox_inches="tight",
    )
    plt.close()

    # plt.figure(figsize=(6, 0.5 * n_groups / 2.54 + 1))
    plt.figure()

    plot_critical_difference_diagram(
        mean_ranks.droplevel(hue_col),
        pvalue_crosstable,
        crossbar_props={"marker": "."},
    )
    plt.title(title, wrap=True)
    set_axes_size(6, 0.5 * n_groups / 2.54 + 1)
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

    if hue_col is None:
        plot_insignificance_bars(
            positions=positions,
            sig_matrix=pvalue_crosstable,
        )
    else:
        ystart = ax.get_ylim()[1]
        for _, hue_group in data.groupby(hue_col)[estimator_col]:
            # Some groups are dropped by iter_posthoc_comparisons due to missing folds
            hue_group = list(set(hue_group) & set(pvalue_crosstable.index))

            plot_insignificance_bars(
                positions=positions,
                sig_matrix=pvalue_crosstable.loc[hue_group, hue_group],
                ystart=ystart,
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

    # Annotate fac on the bottom right:
    plt.annotate(
        # f"×{fac:.0e}",
        f"{fac:.0e}"[1:],
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

    # # Determine estimator hue_col
    # if hue_col is not None:
    #     data[hue_col] = df.loc[data.index, hue_col]
    #     data[hue_col] = data[hue_col].fillna("none")
    #     new_estimator_names = data["estimator"] + sep + data[hue_col].astype(str)
    #     if transpose_hue:
    #         data[hue_col] = data["estimator"]
    #     data["estimator"] = new_estimator_names
    # else:  # hue_col is None
    #     data["hue_col"] = "no_hue"  # HACK
    #     hue_col = "hue_col"

    # Drop duplicated runs (should be ordered by start time)
    dup = data.duplicated(["dataset", "fold", "estimator"], keep="last")
    if dup.any():
        warnings.warn(
            "The following runs were duplicated and will be removed from the"
            f" analysis:\n{data[dup]}"
        )
        data = data[~dup]

    max_estimators_per_dataset = data.groupby("dataset").estimator.nunique().max()

    allsets_data = (
        data
        # Consider only datasets with all the estimators
        .groupby("dataset").filter(
            lambda x: x.estimator.nunique() == max_estimators_per_dataset
        )
    )
    discarded_datasets = set(data.dataset) - set(allsets_data.dataset)
    if discarded_datasets:
        print(
            # raise RuntimeError(
            "The following datasets were not present for all estimators and"
            " will not be considered for rankings across all datasets:"
            f" {discarded_datasets}"
        )

    max_folds_per_estimator = (
        data.groupby(["dataset", "estimator"]).fold.nunique().max()
    )

    allsets_data = (
        allsets_data
        # Consider only estimators with all the CV folds
        .groupby(["dataset", "estimator"]).filter(
            lambda x: x.fold.nunique() == max_folds_per_estimator
        )
    )

    discarded_runs = set(data[["dataset", "estimator"]].itertuples(index=False)) - set(
        allsets_data[["dataset", "estimator"]].itertuples(index=False)
    )
    if discarded_runs:
        print(
            "The following runs were not present for all CV folds and"
            " will not be considered for rankings across all datasets:"
            f" {discarded_runs}"
        )

    allsets_data = (
        allsets_data.set_index(["dataset", "fold", "estimator", "hue"])  # Keep columns
        .groupby(level=["dataset", "fold"])
        .rank(pct=True)  # Rank estimators per fold
        .groupby(level=["dataset", "estimator", "hue"])  # groupby()
        .mean()  # Average ranks across folds for each estimator
        .rename_axis(index=["fold", "estimator", "hue"])  # 'dataset' -> 'fold'
        .reset_index()
        .assign(dataset="all_datasets")
    )

    data = pd.concat([allsets_data, data], ignore_index=True, sort=False)

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

    main_outdir.mkdir(exist_ok=True, parents=True)
    friedman_statistics.to_csv(main_outdir / "test_statistics.tsv", sep="\t")

    data = data.dropna(axis=1, how="all")  # FIXME: something is bringing nans back

    table_lines = []
    grouped = data.groupby("dataset", sort=False)

    # Make visualizations of pairwise estimator comparisons.
    for dataset_name, dataset_group in tqdm(grouped, total=grouped.ngroups):

        # Existence is assured by make_visualizations()
        outdir = main_outdir / dataset_name

        for metric, pvalue_crosstable, mean_ranks in iter_posthoc_comparisons(
            dataset_group,
            y_cols=metric_names,
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
    table.to_csv(main_outdir / "comparison_table.tsv", sep="\t")
    table.to_html(main_outdir / "comparison_table.html", escape=False)
    (
        table.apply(
            lambda x: x.str.replace(r"<b>(.*?)</b>", r"\\textbf{\1}", regex=True)
        ).to_latex(main_outdir / "comparison_table.tex")
    )


@click.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the configuration file.",
)
def main(config):
    """Generate statistical comparisons between run results."""

    config = yaml.safe_load(config.read_text())

    mlflow.set_tracking_uri("mlruns")
    data = mlflow.search_runs(
        filter_string="status = 'FINISHED'",
        order_by=["start_time"],
        search_all_experiments=True,
    )
    breakpoint()
    data = (
        data.dropna(subset=["tags.estimator", "tags.dataset"])
        .drop_duplicates("tags.mlflow.runName", keep="last")
        .set_index(["tags.estimator", "tags.dataset", "tags.validation_setting"])
        .filter(like="metrics.", axis="columns")
    )
    breakpoint()
    data.columns = pd.MultiIndex.from_frame(
        data.columns.str.extract(r"metrics\.(\w+)\.(\d+)")
    )
    data = (
        data.stack(future_stack=True)
        .rename_axis(["estimator", "dataset", "fold"])
        .reset_index()
    )
    data["fold"] = data["fold"].astype(int)

    make_statistical_comparisons(
        data=data,
        estimator_subset=args.estimators,
        dataset_subset=args.datasets,
        metric_subset=args.metrics,
        main_outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
