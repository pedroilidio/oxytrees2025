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


def plot_empirical_complexity_results(*, data: pd.DataFrame, outdir: Path):
    reg_cutoff = 150
    xlabel = "Number of instances in both dimensions"

    # Convert nanoseconds to minutes
    data.loc[:, ["fit_time", "predict_time", "predict_time_single"]] /= 60e9

    plt.subplots(1, 2, figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.grid(axis="y")
    plt.xlabel(xlabel)
    plt.ylabel("Training time (minutes)")

    plt.subplot(1, 2, 2)
    plt.xlabel(xlabel)
    plt.xscale("log")
    plt.yscale("log")

    for estimator, estimator_data in data.groupby("estimator"):
        plt.subplot(1, 2, 1)
        plt.scatter(
            estimator_data.data_size,
            estimator_data.fit_time,
            label=estimator,
            marker="|",
        )

        plt.subplot(1, 2, 2)
        plot = plt.scatter(
            estimator_data.data_size,
            estimator_data.fit_time,
            # np.log2(estimator_data.data_size),
            # np.log2(estimator_data.fit_time),
            # label=estimator,
            marker="|",
        )
        color = plot.get_facecolors()[0]
        xlim = plt.xlim()
        ylim = plt.ylim()

        plt.axvline(reg_cutoff, color="black", linestyle="--")

        reg_data = estimator_data.loc[:, ["data_size", "fit_time"]].dropna()
        reg_data = reg_data.loc[reg_data.data_size > reg_cutoff]

        lr = stats.linregress(
            np.log2(reg_data.data_size).values,
            np.log2(reg_data.fit_time).values,
        )

        x = np.linspace(xlim[0], xlim[1], 5)
        plt.plot(
            x,
            x**lr.slope * 2**lr.intercept,
            "-",
            color=color,
            label=f"{estimator} (${lr.slope:.3g} \pm {lr.stderr:.2g}$)",
        )
        plt.xlim(xlim)
        plt.ylim(ylim)

    # plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.legend(loc="upper left", framealpha=1)

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "training_empirical_complexity"
    plt.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(
        out.with_suffix(".pdf"), bbox_inches="tight", transparent=True, pad_inches=0.02
    )
    plt.close()

    # =====================================

    reg_cutoff = 150
    predict_data = data.loc[:, ["data_size", "predict_time", "predict_time_single"]]
    predict_data = predict_data.dropna()
    predict_data = predict_data.loc[predict_data.predict_time_single < 0.8]

    plt.subplots(1, 2, figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.grid(axis="y")
    plt.ylabel("Prediction time (minutes)")
    plt.xlabel(xlabel)

    plt.subplot(1, 2, 2)
    plt.xlabel(xlabel)
    plt.xscale("log")
    plt.yscale("log")

    for predict_function, new_name in (
        ("predict_time", "Batch"),
        ("predict_time_single", "Individual"),
    ):
        plt.subplot(1, 2, 1)

        plot = plt.scatter(
            predict_data.data_size,
            predict_data[predict_function],
            marker="|",
        )
        color = plot.get_facecolors()[0]

        reg_data = predict_data.loc[
            predict_data.data_size > reg_cutoff, ["data_size", predict_function]
        ]
        lr = stats.linregress(
            np.log2(reg_data.data_size).values,
            np.log2(reg_data[predict_function]).values,
        )

        plt.subplot(1, 2, 2)
        plt.axvline(reg_cutoff, color="black", linestyle="--")
        plt.scatter(
            predict_data.data_size,
            predict_data[predict_function],
            marker="|",
        )
        xlim = plt.xlim()
        ylim = plt.ylim()

        x = np.linspace(xlim[0], xlim[1], 5)
        plt.plot(
            x,
            x**lr.slope * 2**lr.intercept,
            "-",
            color=color,
            label=f"{new_name} (${lr.slope:.3g} \pm {lr.stderr:.2g}$)",
        )
        # plt.axline(
        #     xy1=(0, lr.intercept_),
        #     slope=lr.coef_[0],
        #     color=color,
        #     linestyle="-",
        #     label=f"{new_name} ({lr.coef_[0]:.3g})",
        # )
        plt.xlim(xlim)
        plt.ylim(ylim)

    # plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.legend(loc="upper left", framealpha=1)

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "predicting_empirical_complexity"

    plt.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(
        out.with_suffix(".pdf"), bbox_inches="tight", transparent=True, pad_inches=0.02
    )
    plt.close()


@click.command()
@click.option(
    "--data-path",
    "--data",
    "-d",
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
# @click.option(
#     "--renaming",
#     type=click.Path(dir_okay=False, path_type=Path),
#     help="Path to the YAML file containing name substitutions.",
# )
def main(data_path, outdir):
    data = pd.read_csv(data_path)

    # renaming = renaming and yaml.safe_load(renaming.open())
    # if renaming:
    #     data["estimator"] = data["estimator"].replace(renaming["estimator"])

    # Use only estimators that are present in the renaming dictionary
    data["estimator"] = data["estimator"].map(ESTIMATOR_RENAMING)

    plot_empirical_complexity_results(data=data, outdir=outdir)


if __name__ == "__main__":
    main()
