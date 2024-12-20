from pathlib import Path

import mlflow
import pandas as pd
import click

BASEDIR = Path(__file__).resolve().parents[1]


@click.command()
@click.option("--tracking-uri", default="sqlite:///mlruns.db")
@click.option(
    "--output-directory",
    "--out",
    "-o",
    type=click.Path(
        file_okay=False,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    default=BASEDIR / "results",
)
def main(tracking_uri, output_directory):
    index_cols = [
        "tags.estimator",
        "tags.data_size",
        "tags.iteration",
    ]

    mlflow.set_tracking_uri(tracking_uri)

    print("Gathering finished runs...")
    data = mlflow.search_runs(
        filter_string="status = 'FINISHED'",
        order_by=["start_time"],
        experiment_names=["empirical_complexity_analysis"],
        max_results=50_000,  # Max allowed by MLflow
    )
    data = (
        data.dropna(subset=index_cols)
        .drop_duplicates(index_cols, keep="last")
        .set_index(index_cols)
        .filter(like="metrics.", axis="columns")
        .reset_index()
    )
    data.columns = data.columns.str.split(".", n=1).str[1]

    print("Saving results...")
    out_fit = output_directory / "empirical_complexity_fit.csv"
    out_fit.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_fit, index=False)
    print(f"Results saved to {out_fit}")


    # ================================================================================

    index_cols = [
        "tags.data_size",
        "tags.iteration",
    ]

    mlflow.set_tracking_uri(tracking_uri)

    print("Gathering finished runs...")
    data = mlflow.search_runs(
        filter_string="status = 'FINISHED'",
        order_by=["start_time"],
        experiment_names=["empirical_complexity_predict"],
        max_results=50_000,  # Max allowed by MLflow
    )
    data = (
        data.dropna(subset=index_cols)
        .drop_duplicates(index_cols, keep="last")
        .set_index(index_cols)
        .filter(like="metrics.", axis="columns")
        .reset_index()
    )
    data.columns = data.columns.str.split(".", n=1).str[1]

    print("Saving results...")
    out_pred = output_directory / "empirical_complexity_predict.csv"
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_pred, index=False)
    print(f"Results saved to {out_pred}")


if __name__ == "__main__":
    main()
