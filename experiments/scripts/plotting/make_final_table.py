from pathlib import Path
import warnings

import pandas as pd
import yaml
import click
import seaborn as sns
import matplotlib.pyplot as plt


# TODO: move to separate file
RENAME_VALIDATION_SETTING = {
    "TT": "0\%",
    "TT_25": "25\%",
    "TT_50": "50\%",
    "TT_75": "75\%",
}


def set_rc_params():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.weight": "bold",
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "legend.framealpha": 1,
            # "axes.titlesize": 10,
            # "font.serif": ["Times New Roman"],
            # "font.size": 10,
            # "xtick.labelsize": 10,
            # "ytick.labelsize": 10,
            # "legend.fontsize": 10,
            # "legend.title_fontsize": 10,
            # "axes.linewidth": 1.5,
            # "lines.linewidth": 1.5,
            # "lines.markersize": 5,
            # "xtick.major.width": 1.5,
            # "ytick.major.width": 1.5,
            # "xtick.minor.width": 1.5,
            # "ytick.minor.width": 1.5,
            # "xtick.major.size": 5,
            # "ytick.major.size": 5,
            # "xtick.minor.size": 3,
            # "ytick.minor.size": 3,
            # "legend.frameon": False,
            # "legend.loc": "upper right",
        }
    )


def plot_masking_vs_score(data: pd.DataFrame, outdir: Path):
    data = data.groupby(
        level=["dataset", "estimator", "validation_setting", "metric"]
    ).mean()

    transductive_mask = (
        data.index.get_level_values("metric").str.endswith("(Transductive)")
    )

    # Divide all validation_settings by the 0% PMP (25% in the transductive case)
    denom = pd.concat(
        [
            data.loc[pd.IndexSlice[:, :, ["0\\%"]]],
            data.loc[pd.IndexSlice[:, :, ["25\\%"], transductive_mask]],
        ]
    ).droplevel("validation_setting")

    reldata = (data / denom).dropna()
    reldata = reldata.sort_index(level=["validation_setting", "estimator"])

    for metric, g in reldata.groupby(level="metric", sort=False):
        plt.figure(figsize=(4, 4))
        sns.lineplot(
            data=g,
            x="validation_setting",
            y="value",
            hue="estimator",
            style="estimator",
            # errorbar=("ci", False),
            errorbar=None,
            # errorbar="sd",
            # err_style="bars",
            markers=True,
            dashes=True,
            markersize=12,
            markeredgewidth=1.5,
        )
        is_transductive = metric.endswith("(Transductive)")  # HACK
        plt.xlabel("Masking percent")
        plt.ylabel(f"{metric} rel. to {25 if is_transductive else 0}\% PMP")
        plt.legend(title=None)
        plt.grid(axis="y")
        plt.tight_layout()

        outpath = outdir / f"{metric}"
        plt.savefig(outpath.with_suffix(".png"))
        plt.savefig(
            outpath.with_suffix(".pdf"),
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
        )
        print(f"Saved to {outpath}.(pdf|png)")
        plt.clf()


def make_final_table(data: pd.DataFrame, outdir: Path):
    alldata = (
        data.groupby(level=["dataset", "fold", "validation_setting", "metric"])
        .rank(ascending=False)  # rank estimators in each fold
        .groupby(level=["dataset", "estimator", "validation_setting", "metric"])
        .mean()  # average ranks across folds (each dataset has different number of folds)
        .groupby(level=["estimator", "validation_setting", "metric"])
        .mean()  # average ranks across datasets
        .assign(dataset="Overall rank")  # it will behave like a single dataset
        .set_index("dataset", append=True)
        .reorder_levels(["metric", "estimator", "dataset", "validation_setting"])
        .mul(-1)  # HACK to get the best estimator when we call idxmax
    )

    # Order by mean AUROC rank in the inductive setting
    estimator_order = (
        alldata.loc["AUROC (Inductive)"]
        .groupby(level="estimator")
        .mean()
        .sort_values("value", ascending=False)
        .index
    )

    result = data.groupby(
        level=["metric", "estimator", "dataset", "validation_setting"]
    ).mean()  # average across folds

    result = result.sort_index()
    result = pd.concat([alldata, result])
    index_order = result.index.get_level_values("dataset").unique()

    for metric, metric_group in result.groupby(
        level="metric", group_keys=False, sort=False
    ):
        n_estimators = metric_group.index.get_level_values("estimator").nunique()
        metric_group = metric_group.value
        maxes = (
            metric_group.groupby(level=["dataset", "validation_setting"], sort=False)
            .idxmax()
            .dropna()
        )
        ranks = (
            metric_group.groupby(level=["dataset", "validation_setting"], sort=False)
            .rank(ascending=False)
            .apply("{:.0f}".format)
        )

        metric_group = metric_group.abs()  # HACK
        metric_group = metric_group.map("{:.2f}".format) + "(" + ranks + ")"

        metric_group.loc[maxes] = metric_group.loc[maxes].map("\\textbf{{{}}}".format)
        metric_group = metric_group.unstack(level="estimator").droplevel("metric")

        # reorder columns by mean inductive AUROC rank
        metric_group = metric_group.loc[:, estimator_order]

        metric_group = metric_group.rename_axis(
            columns={"estimator": ""},
            index={"dataset": "Dataset", "validation_setting": "Masking"},
        )

        metric_group.columns = metric_group.columns.str.replace("[", " [")  # HACK

        text_table = metric_group.loc[index_order].to_latex(
            bold_rows=True,
            column_format="ll" + "p{.7cm}" * n_estimators,
            # bold_rows=True, column_format="p{3cm}l" + "p{.7cm}" * n_estimators
        )
        text_table = text_table.replace(f"\\cline{{1-{n_estimators + 2}}}", "\\midrule")
        text_table = text_table.replace("\\midrule\n\\bottomrule", "\\bottomrule")

        # HACK
        metric_prefix = metric.split()[0]
        metric_suffix = metric.split()[1].strip("()")
        if len(metric_suffix) > 2:  # Not TL and LT
            metric_suffix = metric_suffix.lower()

        text_table += (
            "\n\\caption{"
            f"{metric_prefix} in the {metric_suffix}"
            " case. The rank of each value is indicated"
            " within parentheses. The highest score (rank = 1) is presented in"
            " bold. The scores were averaged across the cross-validation folds."
            " The number of folds per dataset is presented by"
            " \\autoref{tab:datasets}. The overall rank was calculated by obtaining"
            " ranks for each fold, then averaging the ranks across folds for each"
            " dataset, and then averaging the results across datasets."
            "}"
        )

        outdir.mkdir(exist_ok=True, parents=True)
        outpath = outdir / f"{metric}.tex"

        with outpath.open("w") as f:
            f.write(text_table)

        print(f"Saved {metric} to {outpath}")


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
    "--output-path",
    "-o",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
)
@click.option(
    "--renaming",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the YAML file containing name substitutions.",
)
def main(config, results_table, output_path, renaming):
    set_rc_params()

    config = yaml.safe_load(config.read_text())
    original_data = pd.concat(map(pd.read_csv, results_table))

    original_data = original_data.sort_values("start_time")

    dup = original_data.duplicated(
        ["dataset", "fold", "estimator", "validation_setting"], keep="last"
    )

    if dup.any():
        warnings.warn(
            "The following runs were duplicated and will be removed from the"
            f" analysis:\n{original_data[dup]}"
        )
        original_data = original_data[~dup]

    original_data = original_data.drop(columns="start_time")

    if renaming:
        renaming = yaml.safe_load(renaming.read_text())
    else:
        renaming = {}

    for i, config_object in enumerate(config):
        if not config_object.get("active", True):
            continue

        estimator_subset = sum(config_object["estimator"], [])

        data = original_data.loc[
            :,
            ["dataset", "estimator", "validation_setting", "fold"]
            + config_object["scoring"],
        ]
        data = data.loc[data.estimator.isin(estimator_subset)]
        data = data.loc[data.dataset.isin(config_object["dataset"])]
        data = data.loc[
            data.validation_setting.isin(config_object["validation_setting"])
        ]
        data = (
            data.set_index(["dataset", "estimator", "validation_setting", "fold"])
            .rename_axis(columns="metric")
            .stack(future_stack=True)
            .rename("value")
        )

        # Create joint semi-inductive data
        semi_inductive_mask = data.index.get_level_values("metric").str.match("LT|TL")
        semi_inductive_data = data.loc[semi_inductive_mask].copy()

        new_folds = (
            semi_inductive_data.index.get_level_values("fold").astype(str)
            + semi_inductive_data.index.get_level_values("metric").str[:2]
        )

        # HACK: keep folds between LT and TL different
        semi_inductive_data = (
            semi_inductive_data.reset_index("fold")
            .assign(fold=new_folds)
            .set_index("fold", append=True)
        )

        new_metric = (
            semi_inductive_data.index.get_level_values("metric")
            .str.replace("TL", "+++++")  # HACK
            .str.replace("LT", "LT+TL")
            .str.replace("+++++", "LT+TL")
        )

        # HACK: rename LT and TL to LT+TL for the joint semi-inductive data
        semi_inductive_data = (
            semi_inductive_data.reset_index("metric")
            .assign(metric=new_metric)
            .set_index("metric", append=True)
        )

        data = pd.concat([data, semi_inductive_data])

        data = data.reset_index()
        for key, mapping in renaming.items():
            data[key] = data[key].replace(mapping)
        data["validation_setting"] = data["validation_setting"].replace(
            RENAME_VALIDATION_SETTING
        )
        data = data.set_index(
            ["dataset", "estimator", "validation_setting", "fold", "metric"]
        )
        data = data.dropna()

        make_final_table(data, outdir=output_path / str(i))
        plot_masking_vs_score(data, outdir=output_path / str(i))


if __name__ == "__main__":
    main()
