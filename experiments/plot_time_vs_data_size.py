import argparse
from pathlib import Path

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset_info(metadata_path):
    return pd.read_table(metadata_path, index_col="dataset_name")


def labeled_regplot(
    data,
    y="log_fit_time",
    x="log_n_samples",
    hue="estimator.name",
    **kwargs,
):
    slope_estimates = {}
    standard_errors = {}
    n = {}

    # Compute linear regression coefficients and their uncertainty
    for estimator, subset in data.groupby(hue):
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            subset[x], subset[y]
        )
        print(f"{estimator}: {slope=:.4f}, {intercept=:.4f}, {std_err=:.2g}")
        slope_estimates[estimator] = slope
        standard_errors[estimator] = std_err
        n[estimator] = subset.shape[0]

        # Plot the results
        sns.regplot(
            data=subset,
            y=y,
            x=x,
            truncate=False,
            marker=".",
            line_kws={
            #"linestyle": linestyle[i],
            },
            scatter_kws={
                #"s": 5,
                # "edgecolor": "white",
                # "zorder": 10,
            },
            # facet_kws={"legend_out": False},
            label=f"{estimator} ({slope = :.4f} $\\pm$ {std_err:.2g})",
            **kwargs,
        )

    return slope_estimates, standard_errors, n


def plot_time_vs_data_size(outdir, metadata_path, results_path):
    print("Loading data...")
    metadata = load_dataset_info(metadata_path)
    data = pd.read_table(results_path)
    data = data.loc[data["dataset.name"].isin(metadata.index)]

    data["n_samples"] = np.sqrt(metadata.loc[data["dataset.name"], "n_interactions"].values)
    # HACK: discard small datasets?
    # data = data.loc[data.n_samples >= 200]

    data["log_n_samples"] = np.log2(data.n_samples)
    data["log_fit_time"] = np.log2(data["results.fit_score_time"])
    # data["log_fit_time"] = np.log2(data["results.fit_time"])
    # data["log_score_time"] = np.log2(data["results.score_time"])

    print("Plotting lines...")
    plt.figure(figsize=(10, 4))
    slope, std, n = labeled_regplot(
        data=data,
        y="log_fit_time",
        x="log_n_samples",
        hue="estimator.name",
    )
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    outdir.mkdir(exist_ok=True, parents=True)
    plt.savefig(outdir / "time_vs_data_size.png")
    print("Saved to", outdir / "time_vs_data_size.png")
    plt.clf()
    
    slope_data = pd.DataFrame({
        "slope": slope,
        "slope_std": std,
        "n": n,
    }).sort_values("slope")

    print("Plotting slopes...")
    plt.figure(figsize=(10, 4))
    plt.errorbar(
        x=slope_data.index,
        y=slope_data.slope,
        yerr=slope_data.slope_std,
        fmt="o",
    )
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(outdir / "slopes_time_vs_data_size.png")
    print("Saved to", outdir / "slopes_time_vs_data_size.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", "-o", type=Path, default=Path("."))
    parser.add_argument(
        "--metadata", "-m", type=Path, required=True,
    )
    parser.add_argument("--results", "-r", type=Path, default=Path("results.tsv"))
    args = parser.parse_args()

    plot_time_vs_data_size(
        outdir=args.outdir,
        metadata_path=args.metadata,
        results_path=args.results,
    )


if __name__ == "__main__":
    main()
