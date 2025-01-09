from pathlib import Path

import numpy as np
import pandas as pd
import click
import matplotlib.pyplot as plt
import yaml
from scipy import stats


# TODO: move to separate file
ESTIMATOR_RENAMING = {
    "bxt_bgso_kronrls": "Oxytree[KronRLS]",
    "bxt_bgso": "Oxytree[Deep]",
    "bxt_gmo": "PBCT[Deep]",
}


def plot_data_size_vs_time(
    *,
    data: pd.DataFrame,
    reg_data: pd.DataFrame,
    data_size_col: str,
    time_col: str,
    group_col: str,
    ylabel: str = "Training time (minutes)",
    xlabel: str = "Data size",
):
    plt.subplots(1, 3, figsize=(18, 4))

    plt.subplot(1, 3, 1)
    plt.grid(axis="y")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.subplot(1, 3, 2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale("log")
    plt.yscale("log")

    plt.subplot(1, 3, 3)
    plt.xlabel(xlabel)
    plt.ylabel("Residuals")
    plt.xscale("log")
    # plt.yscale("log")

    for i, (estimator, estimator_data) in enumerate(data.groupby(group_col)):
        plt.subplot(1, 3, 1)
        plt.scatter(
            estimator_data[data_size_col],
            estimator_data[time_col],
            label=estimator,
            marker="|",
            color=f"C{i}",
        )

        plt.subplot(1, 3, 2)
        plot = plt.scatter(
            estimator_data[data_size_col],
            estimator_data[time_col],
            marker="|",
            color=f"C{i}",
        )

    plt.subplot(1, 3, 2)
    xlim = plt.xlim()
    ylim = plt.ylim()

    for i, (group_name, group_data) in enumerate(reg_data.groupby(group_col)):
        reg_data = group_data.loc[:, [data_size_col, time_col]].dropna()
        lr = stats.linregress(
            np.log2(reg_data[data_size_col]).values,
            np.log2(reg_data[time_col]).values,
        )
        x = np.linspace(xlim[0], xlim[1], 5)

        plt.subplot(1, 3, 2)
        plt.plot(
            x,
            x ** lr.slope * 2 ** lr.intercept,
            "-",
            color=f"C{i}",
            label=f"{group_name} (${lr.slope:.3g} \pm {lr.stderr:.2g}$)",
        )

        residuals = (
            reg_data[time_col] - reg_data[data_size_col] ** lr.slope * 2**lr.intercept
        )
        residual_bins = [np.mean(a) for a in np.array_split(residuals, 20)]
        bin_positions = [np.mean(a) for a in np.array_split(reg_data[data_size_col], 20)]

        plt.subplot(1, 3, 3)
        plt.plot(
            # reg_data[data_size_col],
            # residuals,
            bin_positions,
            residual_bins,
            # np.log2(np.abs(residuals) * 100 + 1) * np.sign(residuals),
            "|",
            color=f"C{i}",
            label=group_name,
        )

    plt.subplot(1, 3, 2)
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.legend(loc="upper left", framealpha=1)


def plot_empirical_complexity_fit(*, data: pd.DataFrame, outdir: Path):
    xlabel = "Number of instances in each dimension"
    reg_cutoff = 200

    # Convert nanoseconds to minutes
    data.loc[:, "fit_time"] /= 60e9
    reg_data = data.loc[data.data_size > reg_cutoff]

    plot_data_size_vs_time(
        data=data,
        reg_data=reg_data,
        data_size_col="data_size",
        time_col="fit_time",
        group_col="estimator",
        xlabel=xlabel,
        ylabel="Training time (minutes)",
    )
    plt.axvline(reg_cutoff, color="black", linestyle="--", label="Regression cutoff")

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "training_empirical_complexity"
    plt.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(
        out.with_suffix(".pdf"), bbox_inches="tight", transparent=True, pad_inches=0.02
    )
    plt.close()


def plot_empirical_complexity_predict(*, data: pd.DataFrame, outdir: Path):
    predict_time_cutoff = 4000
    predict_time_single_cutoff = 400
    renaming = {
        "predict_time": "Oxytree (batch)",
        "predict_time_single": "PBCT (individual)",
    }

    data = data.copy()

    # Convert nanoseconds to minutes
    data.loc[:, ["predict_time", "predict_time_single"]] /= 60e9

    predict_data = data.loc[:, ["data_size", "predict_time", "predict_time_single"]]
    predict_data = predict_data.melt(
        id_vars="data_size",
        value_vars=["predict_time", "predict_time_single"],
        var_name="predict_function",
    )
    predict_data = predict_data.sort_values("predict_function")

    # Apply cuttoffs to the data
    reg_data = predict_data.loc[
        (
            (predict_data.predict_function == "predict_time")
            & (predict_data.data_size > predict_time_cutoff)
        )
        | (
            (predict_data.predict_function == "predict_time_single")
            & (predict_data.data_size > predict_time_single_cutoff)
        )
    ]

    # Rename the predict function
    predict_data.loc[:, "predict_function"] = predict_data.predict_function.replace(
        renaming
    )
    reg_data.loc[:, "predict_function"] = reg_data.predict_function.replace(renaming)

    plot_data_size_vs_time(
        data=predict_data,
        reg_data=reg_data,
        data_size_col="data_size",
        time_col="value",
        group_col="predict_function",
        xlabel="Number of instances in each dimension",
        ylabel="Prediction time (minutes)",
    )

    # Possible because of sorting
    plt.axvline(predict_time_cutoff, color="C0", linestyle="--")
    plt.axvline(predict_time_single_cutoff, color="C1", linestyle="--")

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "predicting_empirical_complexity"

    plt.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(
        out.with_suffix(".pdf"), bbox_inches="tight", transparent=True, pad_inches=0.02
    )
    plt.close()


@click.command()
@click.option(
    "--fit-results-path",
    "-f",
    type=click.Path(
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
        exists=True,
    ),
    required=True,
)
@click.option(
    "--predict-results-path",
    "-p",
    type=click.Path(
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
        exists=True,
    ),
    required=True,
)
@click.option(
    "--outdir",
    "--out",
    "-o",
    type=click.Path(
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
        exists=False,
    ),
    required=True,
)
def main(fit_results_path, predict_results_path, outdir):
    fit_data = pd.read_csv(fit_results_path)
    fit_data["estimator"] = fit_data["estimator"].map(ESTIMATOR_RENAMING)
    plot_empirical_complexity_fit(data=fit_data, outdir=outdir)

    predict_data = pd.read_csv(predict_results_path)
    plot_empirical_complexity_predict(data=predict_data, outdir=outdir)


if __name__ == "__main__":
    main()
