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
import pandas as pd
import yaml
import click
from sklearn.base import clone, BaseEstimator
import sklearn.exceptions
from sklearn.utils.validation import check_is_fitted
from mlflow import MlflowClient
from mlflow.types import TensorSpec, Schema
from mlflow.entities import ViewType
from mlflow.models.signature import ModelSignature
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))  # HACK

from mlflow_utils import log_sklearn_model, get_experiment_id_from_name, uri_to_path

# from run_experiments import apply_estimator
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


class AttributeWrapperMixin:
    """A mixin that allows accessing attributes of the wrapped estimator.

    If an attribute is not found in the current class,
    it will be searched in the meta attributes.
    """

    _meta_attributes = ("estimator_", "estimator")

    # Only called if name is not found in the instance
    def __getattr__(self, name):
        if name in self._meta_attributes:  # FIXME: only works for first meta attribute
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )
        try:
            for meta_attr in self._meta_attributes:
                return getattr(getattr(self, meta_attr), name)
        except AttributeError:
            pass

        type_dict = {
            attr: (
                getattr(self, attr).__class__.__name__
                if hasattr(self, attr)
                else "(Not set!)"
            )
            for attr in self._meta_attributes
        }
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
            f" and nor does its attribute(s): {type_dict}"
        )


# HACK
class IndividualTrees(BaseEstimator, AttributeWrapperMixin):
    """A base class for individual trees in a forest.

    This class is used to wrap individual tree estimators and provide a common interface
    for accessing their attributes.
    """

    _meta_attributes = ("estimator",)

    def __init__(
        self,
        estimator: BaseEstimator,
        individuals_attribute: str = "estimators_",
    ):
        self.estimator = estimator
        self.individuals_attribute = individuals_attribute

    def fit(
        self, X, y, **kwargs
    ):  # FIXME: should not be necessary because of the mixin
        self.estimator.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        # self._last_X = X
        # return self.estimator.predict(X)
        return self.predict_individually(X)

    def predict_individually(self, X):
        individuals = getattr(self.estimator, self.individuals_attribute)
        return [individual.predict(X) for individual in individuals]

    @property
    def estimators_(self):  # FIXME: should not be necessary because of the mixin
        """Return the individual estimators."""
        return self.estimator.estimators_

    # def get_last_individual_predictions(self):
    #     """Get predictions from the last individual tree."""
    #     if not hasattr(self, "_last_X"):
    #         raise RuntimeError("No data has been passed to the estimator yet.")
    #     return self.predict_individually(self._last_X)

    # def __sklearn_is_fitted__(self):  # FIXME: should not be necessary because of the mixin
    #     try:
    #         check_is_fitted(self.estimator)
    #         return True
    #     except sklearn.exceptions.NotFittedError:
    #         return False


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
        dataset: Dataset = self.dataset_loader.load()
        n_trees_updater: NumberOfTreesUpdater = self.estimator_loader
        estimator = n_trees_updater.estimator_

        assert isinstance(
            n_trees_updater.forest, IndividualTrees
        ), f"Expected IndividualTrees, got {type(n_trees_updater.forest)}."

        # n_trees_updater.set_n_trees(max(self.values))
        individual_preds = apply_estimator(estimator, dataset, self.fold)

        scores = []

        # How many diferent order of trees to sample
        n_samples = 50  # TODO: make it configurable
        values_array = np.asarray(self.values, dtype=int)
        max_n_trees = values_array.max()
        rng = np.random.default_rng(0)

        print("Logging metrics...")
        for pred in tqdm(individual_preds, desc="Logging metrics"):
            # pred.predictions.shape == (n_total_trees, *y.shape)
            # permutations.shape == (n_samples, max_n_trees, *y.shape)
            permutations = np.stack(
                [
                    rng.choice(pred.predictions, size=max_n_trees, replace=False)
                    for _ in range(n_samples)
                ],
                axis=0,
            )

            # cumsums_over_trees.shape == (n_samples, max_n_trees, *y.shape)
            cumsums_over_trees = np.cumsum(permutations, axis=1)

            prediction_samples = []
            for n_trees, i_sample in product(values_array, range(n_samples)):
                prediction_samples.append(
                    replace(
                        pred,
                        name=f"{pred.name}__sample{i_sample}",
                        predictions=cumsums_over_trees[i_sample, n_trees - 1],
                        step=n_trees,
                    )
                )
            # Log scores and append them to the scores list.
            scores.extend(self.log_metrics(prediction_samples))

        # # HACK
        # scores_df = pd.DataFrame(scores)
        # scores_df = scores_df.assign(
        #     **scores_df.setting_name.str.extract(
        #         r"^(?P<fold_name>.*?)__sample(?P<sample>\d+)$"
        #     )
        # )
        # mean_scores = (
        #     scores_df.groupby(["fold_name", "metric", "step"])["value"]
        #     .mean()
        #     .reset_index()
        # )

        # for score in tqdm(mean_scores.itertuples(), desc="Logging mean scores"):
        #     self.client.log_metric(
        #         run_id=self._run_id,
        #         key=f"{score.fold_name}__{score.metric}",
        #         value=score.value,
        #         step=score.step,
        #         synchronous=False,
        #     )

        # print("Logging individual predictions...")
        # self.client.log_dict(
        #     self._run_id,
        #     [p.to_json() for p in individual_preds],
        #     "individual_predictions.json",
        # )

        # FIXME: Logging predictions is too heavy. We log only the metrics.
        # return estimator, predictions
        return estimator, []


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

    y_pred = estimator.predict(X_predict)
    print(f"estimator: {estimator}")
    print(f"Predictions shape: {len(y_pred)}, {len(y_pred[0])}")
    print(f"y shape: {dataset.y.shape}")
    y_pred = np.stack([y_el.reshape(dataset.y.shape) for y_el in y_pred], axis=-1)
    print(f"Predictions shape 2: {y_pred.shape}")

    predictions_dataset: Dataset = replace(dataset, y=y_pred)
    predictions_split: DatasetSplit = fold.split_dataset(predictions_dataset)

    predictions_list = []
    for set_name, dataset in dataset_split.__dict__.items():
        y_target = dataset.y
        y_hat = np.moveaxis(
            getattr(predictions_split, set_name).y, (0, 1, 2), (1, 2, 0)
        )
        print(f"Predictions shape partition: {y_hat.shape}")
        mask = ~np.isnan(y_target)  # Mask for non-nan values in the target

        predictions_list.append(
            PredictionRecord(
                name=set_name,
                targets=y_target[mask],
                predictions=y_hat[:, mask],
            )
        )

    return predictions_list


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
        # internal_estimator = self.load()[0]
        internal_estimator = self.estimator_ = self.load()[0]

        if self.forest_param_name is None:
            internal_forest = internal_estimator
            # HACK
            internal_estimator = self.estimator_ = IndividualTrees(internal_forest)
        else:
            internal_forest = internal_estimator.get_params().get(
                self.forest_param_name, None
            )
            self.estimator_.set_params(
                **{self.forest_param_name: IndividualTrees(internal_forest)}  # HACK
            )

        # if internal_forest is None:
        #     raise AttributeError(
        #         f"The estimator {internal_estimator} does not have a"
        #         f" {self.forest_param_name} attribute."
        #     )
        # if not hasattr(internal_forest, self.n_trees_param_name):
        #     raise AttributeError(
        #         f"The forest {internal_forest} does not have a"
        #         f" {self.n_trees_param_name} attribute."
        #     )

    # @functools.cached_property  # Copies
    # @property
    # def estimator_(self) -> BaseEstimator:
    #     if hasattr(self, "_estimator_"):
    #         return self.estimator_
    #     self.estimator_ = self.load()[0]
    #     return self.estimator_

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
        # self.forest.set_params(**{self.n_trees_param_name: n_trees})
        self.forest.estimator.set_params(**{self.n_trees_param_name: n_trees})

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
    "--interval",
    type=int,
    default=10,
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
    interval,
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

    n_trees_values = tuple(range(start, end + 1, interval))  # Inclusive range

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
            finished_runs = {run.info.run_name for run in finished_runs}

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
