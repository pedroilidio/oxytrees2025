from typing import Iterable, Callable
import functools
from dataclasses import dataclass, field, replace
import traceback
import os
import sys
import warnings
from pathlib import Path
from itertools import product
from importlib import import_module
import multiprocessing as mp
from warnings import warn

import numpy as np
import yaml
import click
from sklearn.base import clone, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from mlflow import MlflowClient
from mlflow.types import TensorSpec, Schema
from mlflow.entities import ViewType
from mlflow.models.signature import ModelSignature
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))  # HACK

from mlflow_utils import log_sklearn_model, get_experiment_id_from_name, uri_to_path
from run_experiments import apply_estimator
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


@dataclass(kw_only=True)
class NTreesRunExecutor(BaseRunExecutor):
    """Run executor for experiments that update the number of trees in a forest estimator.
    This executor is used to run experiments that modify the number of trees in a
    forest estimator, such as Random Forest or Gradient Boosting.

    Attributes:
    -----------
    values: Iterable[int]
        The values to use for the number of trees in the forest estimator.

    See BaseRunExecutor for other attributes.
    """

    values: Iterable[int] = tuple(range(1, 501))  # Default range from 1 to 500

    # @override
    def apply_estimator(self) -> tuple[BaseEstimator, list[PredictionRecord]]:
        self.check_run_active()
        dataset: Dataset = self.dataset_loader.load()
        n_trees_updater: NumberOfTreesUpdater = self.estimator_loader
        estimator: BaseEstimator = n_trees_updater.estimator_

        predictions: list[PredictionRecord] = []
        for n_trees in self.values:
            # Update the number of trees in the estimator
            n_trees_updater.set_n_trees(n_trees)
            preds = apply_estimator(estimator, dataset, self.fold)
            self.log_metrics(preds, step=n_trees)
            predictions.extend([replace(p, name=f"{p.name}__{n_trees}") for p in preds])

        self.client.log_dict(
            self._run_id,
            [p.to_json() for p in predictions],
            "individual_predictions.json",
        )
        # HACK: we take change of logging the metrics, so we dont't return predictions.
        # return estimator, predictions
        return estimator, []


@dataclass(kw_only=True)
class NumberOfTreesUpdater(EstimatorLoader):
    """Class to update the number of trees in a forest estimator.

    self.estimator_ is created by calling self.estimator_loader().
    if self.estimator_ is not a forest estimator but wraps a forest instead, one must
    specify the name of parameter pointing to the forest in self.forest_param_name.
    This is useful for metaestimators like pipelines.

    Attributes:
    -------------
    estimator_loader: Callable[[], BaseEstimator]
        A callable that returns an unfitted estimator.
    forest_param_name: str | None
        The name of the parameter in the estimator that points to the forest.
    n_trees_param_name: str
        The name of the parameter that specifies the number of trees in the forest.
        Defaults to "n_estimators".
    """

    forest_param_name: str | None
    n_trees_param_name: str = "n_estimators"

    def __post_init__(self):
        """Check if the estimator and forest have the required attributes.

        This checks presence of required attributes in the estimator and forest without
        storing them in the instance yet: we want to check during instantiation in the
        main process and only create and store the estimator in the worker process
        (avoiding pickling the estimator when creating the worker).
        """
        internal_estimator = self.load()[0]

        if self.forest_param_name is None:
            internal_forest = internal_estimator
        else:
            internal_forest = internal_estimator.get_params().get(
                self.forest_param_name, None
            )

        if internal_forest is None:
            raise AttributeError(
                f"The estimator {internal_estimator} does not have a"
                f" {self.forest_param_name} attribute."
            )
        if not hasattr(internal_forest, self.n_trees_param_name):
            raise AttributeError(
                f"The forest {internal_forest} does not have a"
                f" {self.n_trees_param_name} attribute."
            )

    @functools.cached_property  # Copies
    def estimator_(self) -> BaseEstimator:
        if hasattr(self, "_estimator_"):
            return self.estimator_
        return self.load()[0]

    @functools.cached_property
    def import_path_(self) -> str:  # HACK
        return self.load()[1]

    @property
    def forest(self) -> BaseEstimator:
        """Return the forest estimator to update."""
        if self.forest_param_name is None:
            return self.estimator_
        return self.estimator_.get_params()[self.forest_param_name]

    def set_n_trees(self, n_trees: int) -> None:
        """Set the number of trees in the forest."""
        self.forest.set_params(**{self.n_trees_param_name: n_trees})

        current_value = getattr(self.forest, self.n_trees_param_name)
        if current_value != n_trees:
            raise ValueError(
                f"Failed to set {self.n_trees_param_name} to {n_trees}."
                f" Current value: {current_value}."
            )


def parse_estimators(config: dict, code_paths: list[str]) -> list[NumberOfTreesUpdater]:
    """Create estimator loaders from the configuration dictionary.

    Args:
        config: A dictionary containing estimator definitions.

    Returns:
        A dictionary mapping estimator names to NumberOfTreesUpdater instances.
    """
    estimator_loaders = []
    for name, definition in config.items():
        if not isinstance(definition, dict):
            raise ValueError(f"Estimator definition for {name} must be a dict.")
        estimator_loaders.append(
            NumberOfTreesUpdater(
                name=name,
                object_path=definition["object_path"],
                forest_param_name=definition.get("forest_param_name"),
                n_trees_param_name=definition.get("n_trees_param_name", "n_estimators"),
                code_paths=definition.get("code_paths", []) + code_paths,
            )
        )
    return estimator_loaders


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
    help="YAML file with experiment definitions.",
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
    "--start",
    type=int,
    default=1,
    help="Number of trees to start with.",
)
@click.option(
    "--end",
    type=int,
    default=500,
    help="Number of trees to end with.",
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
    start,
    end,
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

    n_trees_values = tuple(range(start, end + 1))  # Inclusive range

    estimator_config = yaml.safe_load(estimator_definitions)
    dataset_config = yaml.safe_load(dataset_definitions)
    fold_config = yaml.safe_load(fold_definitions)
    experiments = yaml.safe_load(experiment_definitions)
    scoring_config = yaml.safe_load(scoring_definitions)

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
                run_view_type=ViewType.ACTIVE_ONLY,
                # Search all experiments. Sometimes we create a different experiment
                # to be able to change the artifact location.
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
                object_path=scoring_config[scoring_name],
                code_paths=list(code_path),
            )
            for scoring_name in experiment_data["scoring"]
        ]
        print("Loading dataset information...")
        dataset_loaders = [
            DatasetLoader(name=dataset_name, **dataset_config[dataset_name])
            for dataset_name in experiment_data["dataset"]
        ]
        print("Loading estimator information...")
        estimators_subset = {
            k: v
            for k, v in estimator_config.items()
            if k in experiment_data["estimator"]
        }
        estimator_loaders = parse_estimators(estimators_subset, list(code_path))

        print(f"Scheduling experiment {experiment_name}...")

        for validation_setting, dataset_loader, estimator_loader in product(
            experiment_data["validation_setting"], dataset_loaders, estimator_loaders
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
                    fold_config.get(dataset_loader.name, []).get(validation_setting, [])
                )
            )
            for fold in folds:
                run_executor = NTreesRunExecutor(
                    client=client,
                    experiment_id=experiment_id,
                    estimator_loader=estimator_loader,
                    metric_function_loaders=metric_function_loaders,
                    dataset_loader=dataset_loader,
                    fold=fold,
                    skip_finished=skip_finished,
                    values=n_trees_values,
                )
                if skip_finished and run_executor.name in finished_runs:
                    warnings.warn(f"Skipping finished run: {run_executor.name}")
                    continue

                jobs.append(
                    pool.apply_async(
                        run_executor, error_callback=run_executor.error_callback
                    )
                )

    print("Running...")
    try:
        pool.close()
        for item in jobs:
            # Without a timeout, we can't interrupt.
            item.wait(timeout=60 * 60 * 24 * 20)  # 20 days timeout
        # pool.join()  # Use instead?
    except:
        pool.terminate()
        raise
    finally:
        # pool.join()  # Correct?
        print("Finished.")


if __name__ == "__main__":
    main()
