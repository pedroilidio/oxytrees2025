# TODO: use mlflow Dataset API
import re
from dataclasses import replace
import traceback
import os
import sys
import warnings
from pathlib import Path
from itertools import product
import multiprocessing as mp
from warnings import warn

import numpy as np
import yaml
import click
from sklearn.base import clone, BaseEstimator
from mlflow import MlflowClient
from mlflow.types import TensorSpec, Schema
from mlflow.models.signature import ModelSignature
from mlflow.entities import ViewType
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))  # HACK

from mlflow_utils import uri_to_path, get_experiment_id_from_name, log_sklearn_model
from data_model import (
    PythonFunctionLoader,
    EstimatorLoader,
    Dataset,
    DatasetLoader,
    Fold,
    DatasetSplit,
    PredictionRecord,
    BaseRunExecutor,
)


# TODO: include number of features (the one bellow is used only for unfit models)
BIPARTITE_SIGNATURE = ModelSignature(
    inputs=Schema([TensorSpec(type=np.dtype("float32"), shape=(2, -1, -1))]),
    outputs=Schema([TensorSpec(type=np.dtype("float64"), shape=(-1,))]),
)


class RunExecutor(BaseRunExecutor):
    # @override
    def apply_estimator(self) -> tuple[BaseEstimator, list[PredictionRecord]]:
        estimator, estimator_code_paths = self.estimator_loader.load()
        dataset: Dataset = self.dataset_loader.load()
        return estimator, apply_estimator(estimator, dataset, self.fold)


def apply_estimator(
    estimator: BaseEstimator, dataset: Dataset, fold: Fold
) -> list[PredictionRecord]:

    # FIXME: should be specified by fold, but requires changing script that generates
    # yaml configs for folds.
    missing_dyads = np.flatnonzero(dataset.y == 0)
    fold = replace(fold, missing_dyads=missing_dyads)

    dataset_split: DatasetSplit = fold.split_dataset(dataset)

    # Test dyads were replaced by nan. Now we replace them by zeros.
    y_LD = np.nan_to_num(dataset_split.LD.y, nan=0.0, copy=True)

    estimator.fit(dataset_split.LD.X, y_LD)

    if dataset.pairwise:
        # If pairwise, we need to drop the test rows/cols from the features.
        X_predict = [
            np.delete(dataset.X[0], fold.test_rows, axis=1),
            np.delete(dataset.X[1], fold.test_cols, axis=1),
        ]
    else:
        X_predict = dataset.X

    y_pred = estimator.predict(X_predict).reshape(dataset.y.shape)

    predictions_dataset: Dataset = replace(dataset, y=y_pred)
    predictions_split: DatasetSplit = fold.split_dataset(predictions_dataset)

    predictions_list = []
    for set_name, dataset in dataset_split.__dict__.items():
        y_target = dataset.y
        y_hat = getattr(predictions_split, set_name).y

        predictions_list.append(
            PredictionRecord(
                name=set_name,
                targets=y_target[~np.isnan(y_target)],
                predictions=y_hat[~np.isnan(y_hat)],
            )
        )

    return predictions_list

@click.command()
@click.option(
    "--estimator-definitions",
    type=click.File("r"),
    required=True,
    help="YAML file with estimator definitions.",
)
@click.option(
    "--dataset-definitions",
    type=click.File("r"),
    required=True,
    help="YAML file with dataset definitions.",
)
@click.option(
    "--fold-definitions",
    type=click.File("r"),
    required=True,
    help="YAML file with fold definitions.",
)
@click.option(
    "--experiment-definitions",
    type=click.File("r"),
    required=True,
    help="YAML file with experimentdefinitions.",
)
@click.option(
    "--scoring-definitions",
    type=click.File("r"),
    required=True,
    help="YAML file with scoring definitions.",
)
@click.option(
    "--tracking-uri",
    default="sqlite:///mlruns.db",
    help="MLflow tracking URI.",
)
@click.option(
    "--n-jobs",
    type=int,
    default=1,
    help=(
        "Number of parallel jobs. Set to 0 to use all available cores, set to a"
        " negative value to use all but the specified number of cores."
    ),
)
@click.option(
    "--code-path",
    type=click.Path(file_okay=False, path_type=Path),
    multiple=True,
    help=(
        "Path to directory from where unfitted estimators and scoring functions can be"
        " imported."
    ),
)
@click.option(
    "--skip-finished",
    is_flag=True,
    help="Skip experiments that have already been run.",
)
def main(
    estimator_definitions,
    dataset_definitions,
    fold_definitions,
    experiment_definitions,
    scoring_definitions,
    tracking_uri,
    n_jobs,
    code_path,
    skip_finished=False,
):
    if n_jobs < 1:
        n_jobs = max(1, mp.cpu_count() + n_jobs)

    # Avoid parallelism in the backend side of some libraries.
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[var] = "1"

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

    sys.path.extend(map(str, code_path))  # HACK

    client = MlflowClient(tracking_uri=tracking_uri)
    pool = mp.Pool(n_jobs)

    estimators = yaml.safe_load(estimator_definitions)
    datasets = yaml.safe_load(dataset_definitions)
    fold_definitions = yaml.safe_load(fold_definitions)
    experiments = yaml.safe_load(experiment_definitions)
    scoring_definitions = yaml.safe_load(scoring_definitions)

    jobs = []

    for experiment_name, experiment_data in experiments.items():
        if experiment_data["active"] is False:
            warnings.warn(f"Skipping inactive experiment_data: {experiment_name}")
            continue

        experiment_id = get_experiment_id_from_name(
            client=client,
            experiment_name=experiment_name,
            description=experiment_data["description"],
        )
        experiment = client.get_experiment(experiment_id)
        try:
            path = uri_to_path(experiment.artifact_location)
        except ValueError:
            print(f"Remote artifact location: {experiment.artifact_location}")
        else:
            if not os.access(path, os.W_OK):
                raise RuntimeError(
                    f"Artifact location is not writable: {experiment.artifact_location}"
                )

        if skip_finished:
            # We will check before running as well, since multiple machines may run at
            # the same time and finish new runs in the meantime. This first bulk check
            # is to already rule out all previous runs, it's much faster than checking
            # in each run.
            # NOTE: Make sure to clean up the status = 'RUNNING' runs before running
            # this script.
            print("Collecting finished runs...")
            finished_runs = client.search_runs(
                # filter_string="status = 'FINISHED'",
                filter_string="status != 'FAILED'",
                # Search all experiments. Sometimes we create a different experiment
                # to be able to change the artifact location.
                run_view_type=ViewType.ACTIVE_ONLY,
                experiment_ids=[e.experiment_id for e in client.search_experiments()],
                max_results=50_000,  # Maximum allowed by MLflow
            )
            finished_runs = {
                run.info.run_name
                for run in tqdm(finished_runs, desc="Processing finished runs")
            }

        print("Loading metric functions...")
        metric_function_loaders = [
            PythonFunctionLoader(
                name=scoring_name,
                object_path=scoring_definitions[scoring_name],
                code_paths=list(code_path),
            )
            for scoring_name in experiment_data["scoring"]
        ]
        print("Loading dataset information...")
        dataset_loaders = [
            DatasetLoader(name=dataset_name, **datasets[dataset_name])
            for dataset_name in experiment_data["dataset"]
        ]
        print("Loading estimator information...")
        estimators_loaders = [
            EstimatorLoader(
                name=estimator_name,
                object_path=estimators[estimator_name],
                code_paths=list(code_path),
            )
            for estimator_name in experiment_data["estimator"]
        ]

        print(f"Scheduling experiment {experiment_name}...")

        for validation_setting, dataset_loader, estimator_loader in product(
            experiment_data["validation_setting"], dataset_loaders, estimators_loaders
        ):
            # FIXME: Change the format of fold configuration
            folds = (
                Fold(
                    name=f"{validation_setting}__{fold_index}",
                    test_rows=fold["test_rows"],
                    test_cols=fold["test_cols"],
                    test_dyads=fold["masked_positives"],
                )
                for fold_index, fold in enumerate(
                    fold_definitions.get(dataset_loader.name, []).get(
                        validation_setting, []
                    )
                )
            )
            for fold in folds:
                run_executor = RunExecutor(
                    client=client,
                    experiment_id=experiment_id,
                    estimator_loader=estimator_loader,
                    metric_function_loaders=metric_function_loaders,
                    dataset_loader=dataset_loader,
                    fold=fold,
                    skip_finished=skip_finished,
                )
                if skip_finished and run_executor.name in finished_runs:
                    warnings.warn(f"Skipping finished run: {run_executor.name}")
                    continue

                jobs.append(
                    pool.apply_async(
                        run_executor,
                        error_callback=run_executor.error_callback,
                    )
                )

    print("Running...")
    try:
        pool.close()
        pool.join()
    except:
        pool.terminate()
        raise
    finally:
        print("Finished.")


if __name__ == "__main__":
    main()
