from pathlib import Path
import warnings

import pandas as pd
import yaml
import click


# TODO: move to separate file
RENAME_VALIDATION_SETTING = {
    "TT": "0\%",
    "TT_25": "25\%",
    "TT_50": "50\%",
    "TT_75": "75\%",
}


# TODO: move to separate file
def combine_LT_TL(original_data):
    data = original_data.copy()
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




def make_final_table(data: pd.DataFrame, outdir: Path):
    result = data.groupby(
        level=["dataset", "estimator", "validation_setting", "metric"]
    ).mean()
    for metric, metric_group in result.groupby(level="metric", group_keys=False):
        n_estimators = metric_group.index.get_level_values("estimator").nunique()

        metric_group = metric_group.value
        maxes = (
            metric_group.groupby(level=["dataset", "validation_setting"])
            .idxmax()
            .dropna()
        )
        ranks = (
            metric_group.groupby(level=["dataset", "validation_setting"])
            .rank(ascending=False)
            .apply("{:.0f}".format)
        )

        metric_group = metric_group.map("{:.3f}".format) + "(" + ranks + ")"

        metric_group.loc[maxes] = metric_group.loc[maxes].map("\\textbf{{{}}}".format)
        metric_group = metric_group.unstack(level="estimator").droplevel("metric")

        metric_group = metric_group.rename_axis(
            columns={"estimator": ""},
            index={"dataset": "Dataset", "validation_setting": "Masking"},
        )

        metric_group.columns = metric_group.columns.str.replace("[", " [")  # HACK

        text_table = metric_group.to_latex(
            bold_rows=True, column_format="ll" + "p{.7cm}" * n_estimators
        )
        text_table = text_table.replace(f"\\cline{{1-{n_estimators + 2}}}", "\\midrule")

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
    config = yaml.safe_load(config.read_text())
    original_data = pd.concat(map(pd.read_csv, results_table))

    # Drop duplicated runs (should be ordered by start time)
    dup = original_data.duplicated(
        ["dataset", "fold", "estimator", "validation_setting"], keep="last"
    )
    if dup.any():
        warnings.warn(
            "The following runs were duplicated and will be removed from the"
            f" analysis:\n{original_data[dup]}"
        )
        original_data = original_data[~dup]

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
            ["dataset", "estimator", "validation_setting"] + config_object["scoring"],
        ]
        data = data.loc[data.estimator.isin(estimator_subset)]
        data = data.loc[data.dataset.isin(config_object["dataset"])]
        data = data.loc[
            data.validation_setting.isin(config_object["validation_setting"])
        ]
        data = (
            data.set_index(["dataset", "estimator", "validation_setting"])
            .rename_axis(columns="metric")
            .stack(future_stack=True)
            .rename("value")
        )

        data = data.reset_index()
        for key, mapping in renaming.items():
            data[key] = data[key].replace(mapping)
        data["validation_setting"] = data["validation_setting"].replace(
            RENAME_VALIDATION_SETTING
        )
        data = data.set_index(["dataset", "estimator", "validation_setting", "metric"])
        data = data.dropna()

        make_final_table(data, outdir=output_path / str(i))


if __name__ == "__main__":
    main()
