from pathlib import Path

import numpy as np
import pandas as pd
import click
import matplotlib.pyplot as plt


def plot_empirical_complexity_results(*, data: pd.DataFrame, outdir: Path):
    plt.subplots()
    for estimator, estimator_data in data.groupby("estimator"):
        plt.subplot(1, 2, 1)
        plt.scatter(
            estimator_data.data_size,
            estimator_data.fit_time,
            label=estimator,
        )
        plt.subplot(1, 2, 2)
        plt.scatter(
            np.log2(estimator_data.data_size),
            np.log2(estimator_data.score_time),
            label=estimator,
        )
    out = outdir / "training_empirical_complexity"
    plt.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
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
def main(data_path, outdir):
    data = pd.read_csv(data_path)
    plot_empirical_complexity_results(data=data, outdir=outdir)


if __name__ == "__main__":
    main()
