from pathlib import Path

import mlflow
import pandas as pd
import click

BASEDIR = Path(__file__).resolve().parents[1]


@click.command()
@click.option("--tracking-uri", default="mlruns")
@click.option(
    "--output-path",
    "--out",
    "-o",
    type=click.Path(
        dir_okay=False,
        writable=True,
        resolve_path=True,
        path_type=Path,
        exists=False,
    ),
    default=BASEDIR / "results/results.csv",
)
def main(tracking_uri, output_path):
    index_cols = [
        "tags.estimator",
        "tags.dataset",
        "tags.fold_index",
    ]

    mlflow.set_tracking_uri(tracking_uri)

    print("Gathering finished runs...")
    data = mlflow.search_runs(
        filter_string="status = 'FINISHED'",
        order_by=["start_time"],
        search_all_experiments=True,
    )
    # HACK
    data["tags.dataset"] += "__" + data["tags.validation_setting"]

    data = (
        data.dropna(subset=index_cols)
        .drop_duplicates(index_cols, keep="last")
        .set_index(index_cols)
        .filter(like="metrics.", axis="columns")
    )
    data.columns = data.columns.str.replace("metrics.", "")
    data = data.rename_axis(
        ["estimator", "dataset", "fold"], axis="index"
    ).reset_index()

    print("Saving results...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
