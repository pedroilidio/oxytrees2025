import re
import functools
from dataclasses import dataclass, field, replace
import traceback
from typing import Any, Callable
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
from mlflow import MlflowClient
from mlflow.types import TensorSpec, Schema
from mlflow.models.signature import ModelSignature
from tqdm import tqdm


# TODO: dependency injection for folds, dataset splits, etc.

# TODO: include number of features (the one bellow is used only for unfit models)
BIPARTITE_SIGNATURE = ModelSignature(
    inputs=Schema([TensorSpec(type=np.dtype("float32"), shape=(2, -1, -1))]),
    outputs=Schema([TensorSpec(type=np.dtype("float64"), shape=(-1,))]),
)


class RunException(RuntimeError):
    def __init__(self, original_exception: Exception, run_executor: "RunExecutor"):
        self.original_exception = original_exception
        self.run_executor = run_executor

    def __str__(self):
        return f"RunException: {self.original_exception}"

    def __reduce__(self):  # For pickling support
        return (self.__class__, (self.original_exception, self.run_executor))


@dataclass(kw_only=True, frozen=True)
class Dataset:
    name: str
    X: list[np.ndarray]
    y: np.ndarray
    # Wether or not the X matrices represent pairwise kernels.
    # True is more conservative, hence the default.
    pairwise: bool = True
    missing_indicator: float = 0.0  # Value to use for missing entries in the y matrix.


@dataclass(kw_only=True, frozen=True)
class DatasetLoader:
    name: str
    X: list[Path]
    y: Path
    pairwise: bool = True

    def __post_init__(self):
        valid_args = dict(
            name=str(self.name),
            X=[Path(x).resolve() for x in self.X],
            y=Path(self.y).resolve(),
            pairwise=bool(self.pairwise),
        )
        for key, value in valid_args.items():
            object.__setattr__(self, key, value)

    def load(self) -> Dataset:
        return Dataset(
            name=self.name,
            X=[np.load(x).astype(np.float32, copy=False) for x in self.X],
            y=np.load(self.y).astype(np.float64, copy=False),
            pairwise=self.pairwise,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "X": [str(x) for x in self.X],
            "y": str(self.y),
            "pairwise": self.pairwise,
        }


@dataclass(kw_only=True, frozen=True)
class DatasetSplit:
    TT: Dataset  # Inductive test set: test rows and test columns.
    TL: Dataset  # Row-inductive test set: test rows and train columns.
    LT: Dataset  # Column-inductive test set: train rows and test columns.
    LL: Dataset  # (unmasked) Learning set: train rows and train columns.
    LD: Dataset  # Learning dyads: Zeros and unmasked ones.
    TD: Dataset  # Test dyads: Zeros and masked ones.


@dataclass(kw_only=True, frozen=True)
class Fold:
    name: str
    test_rows: np.ndarray[int]
    test_cols: np.ndarray[int]
    test_dyads: np.ndarray[int] = field(default_factory=list)
    # Dyads that are considered missing are used both in the learning and test sets.
    # For positive-unlabeled learning, these are the positions of zeros in the LL set.
    missing_dyads: np.ndarray[int] = field(default_factory=list)

    def __post_init__(self):
        valid_args = dict(
            name=str(self.name),
            test_rows=np.asarray(self.test_rows, dtype="int64"),
            test_cols=np.asarray(self.test_cols, dtype="int64"),
            test_dyads=np.asarray(self.test_dyads, dtype="int64"),
            missing_dyads=np.asarray(self.missing_dyads, dtype="int64"),
        )
        for key, value in valid_args.items():
            object.__setattr__(self, key, value)

    def to_json(self) -> dict[str, str | list[int]]:
        return {
            "name": self.name,
            "test_rows": self.test_rows.tolist(),
            "test_cols": self.test_cols.tolist(),
            "test_dyads": self.test_dyads.tolist(),
            "missing_dyads": self.missing_dyads.tolist(),
        }

    def split_dataset(self, dataset: Dataset) -> DatasetSplit:
        (X_rows, X_cols), y = dataset.X, dataset.y
        train_rows = np.setdiff1d(np.arange(X_rows.shape[0]), self.test_rows)
        train_cols = np.setdiff1d(np.arange(X_cols.shape[0]), self.test_cols)

        X_rows_test = X_rows[self.test_rows, :]
        X_rows_train = X_rows[train_rows, :]

        X_cols_test = X_cols[self.test_cols, :]
        X_cols_train = X_cols[train_cols, :]

        if dataset.pairwise:
            # If the dataset is pairwise, we need to exclude kernels calculated on the
            # test rows/cols.
            X_rows_test = X_rows_test[:, train_rows]
            X_rows_train = X_rows_train[:, train_rows]

            X_cols_test = X_cols_test[:, train_cols]
            X_cols_train = X_cols_train[:, train_cols]

        TT = replace(
            dataset,
            name=f"{dataset.name}_{self.name}_TT",
            X=[X_rows_test, X_cols_test],
            y=y[self.test_rows, :][:, self.test_cols],
        )
        TL = replace(
            dataset,
            name=f"{dataset.name}_{self.name}_TL",
            X=[X_rows_test, X_cols_train],
            y=y[self.test_rows, :][:, train_cols],
        )
        LT = replace(
            dataset,
            name=f"{dataset.name}_{self.name}_LT",
            X=[X_rows_train, X_cols_test],
            y=y[train_rows, :][:, self.test_cols],
        )
        LL = replace(
            dataset,
            name=f"{dataset.name}_{self.name}_LL",
            X=[X_rows_train, X_cols_train],
            y=y[train_rows, :][:, train_cols],
        )

        y_LD = LL.y.copy()
        y_LD.flat[self.test_dyads] = np.nan  # Remove test dyads from learning set

        # Learning dyads. Ones not in test_dyads, and zeros.
        LD = replace(LL, name=f"{dataset.name}_{self.name}_LD", y=y_LD)

        # All ones in LL
        non_missing_dyads = np.setdiff1d(np.arange(LL.y.size), self.missing_dyads)
        # Ones in LD
        non_missing_train_dyads = np.setdiff1d(non_missing_dyads, self.test_dyads)

        y_TD = LL.y.copy()
        y_TD.flat[non_missing_train_dyads] = np.nan

        # Test dyads. Ones in test_dyads, and zeros.
        TD = replace(LL, name=f"{dataset.name}_{self.name}_TD", y=y_TD)

        return DatasetSplit(LL=LL, LD=LD, TD=TD, LT=LT, TL=TL, TT=TT)


@dataclass(kw_only=True, frozen=True)
class PythonFunctionLoader:
    name: str
    object_path: str
    code_paths: list[Path] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        valid_args = dict(
            name=str(self.name),
            object_path=str(self.object_path),
            code_paths=[Path(p).resolve() for p in self.code_paths],
            params=self.params,
        )
        for key, value in valid_args.items():
            object.__setattr__(self, key, value)

    def load_object(self) -> tuple[Callable, str]:
        module_name, obj_name = self.object_path.rsplit(".", 1)

        sys.path.extend(str(p) for p in self.code_paths)  # HACK
        module = import_module(module_name)
        sys.path = sys.path[: len(sys.path) - len(self.code_paths)]

        return getattr(module, obj_name), str(module.__file__)

    def load(self) -> tuple[Any, str]:
        func, code_path = self.load_object()
        if callable(func):
            return functools.partial(func, **self.params), code_path
        elif self.params:
            warn("Ignoring parameters for non-callable object")
        return func, code_path


# TODO: we are not using params. PythonFunctionLoader is sufficient
class EstimatorLoader(PythonFunctionLoader):
    def load(self) -> tuple[BaseEstimator, str]:
        getter_func, code_path = super().load()
        if callable(getter_func):
            estimator: BaseEstimator = getter_func()
        else:
            estimator: BaseEstimator = clone(getter_func)
        estimator.set_params(**self.params)
        return estimator, code_path


@dataclass(kw_only=True)
class RunExecutor:
    client: MlflowClient
    experiment_id: str
    estimator_loader: EstimatorLoader
    metric_function_loaders: list[PythonFunctionLoader]
    dataset_loader: DatasetLoader
    fold: Fold
    skip_finished: bool = True
    name_tag_order = ("estimator", "dataset", "fold_name")

    def __post_init__(self):
        for tag in self.name_tag_order:
            if tag not in self.tags:
                raise ValueError(f"Missing tag: {tag}")

    @functools.cached_property
    def tags(self) -> dict[str, str]:
        return {
            "estimator": self.estimator_loader.name,
            "dataset": self.dataset_loader.name,
            "fold_name": self.fold.name,  # e.g. LT_75__6
            # HACK: only for backward compatibility:
            "validation_setting": self.fold.name.rsplit("__", 1)[0],  # e.g. LT_75
            "fold_index": self.fold.name.rsplit("__", 1)[1],  # e.g. 6
        }

    @property
    def name(self) -> str:
        return "__".join(self.tags[tag] for tag in self.name_tag_order)

    @staticmethod
    def apply_estimator(
        estimator: BaseEstimator, dataset: Dataset, fold: Fold
    ) -> tuple[BaseEstimator, dict[str, dict[str, list[float]]]]:
        return apply_estimator(estimator, dataset, fold)

    def execute(self):
        try:
            self._execute()
        except KeyboardInterrupt:
            self.client.set_terminated(self.run_id, status="FAILED")
            raise
        except Exception as e:
            raise RunException(original_exception=e, run_executor=self) from e

    def _execute(self):
        if self.skip_finished:
            print(f"Checking if run {self.name} was already executed...")
            finished_runs = self.client.search_runs(
                filter_string=(
                    # f"run_name = '{self.name}' AND status = 'FINISHED'"
                    f"run_name = '{self.name}' AND status != 'FAILED'"
                ),
                experiment_ids=[self.experiment_id],
                max_results=1,
            )
            if finished_runs:
                warn(f"Skipping already finished run: {self.name}")
                return

        self.mlflow_run = self.client.create_run(
            experiment_id=self.experiment_id,
            run_name=self.name,
            tags=self.tags,
        )
        self.run_id = self.mlflow_run.info.run_id

        self.client.log_dict(
            self.run_id, self.dataset_loader.to_json(), "dataset_loader.yml"
        )
        self.client.log_dict(self.run_id, self.fold.to_json(), "fold_definition.yml")

        estimator: BaseEstimator
        estimator, estimator_code_paths = self.estimator_loader.load()
        dataset: Dataset = self.dataset_loader.load()

        # TODO: Log fitted model?
        log_sklearn_model(self.client, self.run_id, estimator, estimator_code_paths)

        fitted_estimator, predictions = self.apply_estimator(
            estimator=estimator,
            dataset=dataset,
            fold=self.fold,
        )
        self.client.log_dict(self.run_id, predictions, "predictions.yml")

        for metric_function_loader in self.metric_function_loaders:
            metric_name = metric_function_loader.name
            metric_func, _ = metric_function_loader.load()

            for setting_name, pred in predictions.items():
                score_name = f"{setting_name}__{metric_name}"
                try:  # Ignore scoring errors
                    score = metric_func(pred["targets"], pred["predictions"])
                    self.client.log_metric(self.run_id, score_name, score)
                except ValueError as e:
                    self.client.log_text(
                        self.run_id, str(e), f"{score_name}__error_message.txt"
                    )
                    self.client.log_text(
                        self.run_id,
                        traceback.format_exc(),
                        f"{score_name}__error_traceback.txt",
                    )

        self.client.set_terminated(self.run_id)

    __call__ = execute


def uri_to_path(uri: str) -> Path:
    """Convert a URI to a Path object."""
    if not re.match(r"^\w+?://", uri):
        return Path(uri).resolve()
    if uri.startswith("file://"):
        # TODO: Python 3.13 has Path.from_uri()
        return Path(uri.removeprefix("file://")).resolve()
    raise ValueError(f"Unsupported URI format: {uri}")


def get_experiment_id_from_name(*, client, experiment_name, description):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Creating experiment: {experiment_name}")
        return client.create_experiment(
            name=experiment_name,
            # artifact_location=None,  # TODO: Specify artifact location
            tags={"mlflow.note.content": description},
        )
    print(f"Found existing experiment: {vars(experiment)}")

    if experiment.lifecycle_stage == "active":
        try:
            path_artifacts = uri_to_path(experiment.artifact_location)
        except ValueError:
            print("Remote artifact location. Using existing experiment.")
            return experiment.experiment_id

        if os.access(path_artifacts, os.W_OK):
            print("Using existing experiment.")
            return experiment.experiment_id

        print(
            f"Artifact_location is not writable: {experiment.artifact_location}"
            f" (Path: {path_artifacts})"
        )

    # HACK
    sep = "__v"
    if sep in experiment_name:
        # If the experiment name already contains a version number, increment it.
        experiment_name, version = experiment_name.rsplit(sep, maxsplit=1)
        try:
            version = str(int(version) + 1)
        except ValueError:
            version += sep + "2"
    else:
        version = "2"

    new_name = experiment_name + sep + version

    print(f"Trying experiment name: {new_name}")
    return get_experiment_id_from_name(
        client=client,
        experiment_name=new_name,
        description=description,
    )


def log_sklearn_model(client, run_id, estimator, code_path):
    # TODO
    # mlflow.models.Model.log(
    #     run_id=run_id,
    #     sk_model=estimator,
    #     artifact_path="model",
    #     flavor=mlflow.sklearn,
    #     signature=BIPARTITE_SIGNATURE,
    #     code_paths=[code_path],
    #     # TODO: Tensors are currently not supported by MLflow
    #     # input_example=[X[0][:2], X[1][:3]],
    # )
    str_params = {k: str(v) for k, v in estimator.get_params().items()}  # FIXME
    client.log_dict(run_id, str_params, "estimator_params.yml")
    client.log_param(run_id, "estimator_class", estimator.__class__.__name__)
    client.log_param(run_id, "estimator_module", code_path)


def apply_estimator(
    estimator: BaseEstimator, dataset: Dataset, fold: Fold
) -> tuple[BaseEstimator, dict[str, dict[str, list[float]]]]:

    estimator = clone(estimator)

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

    predictions: Dataset = replace(dataset, y=y_pred)
    predictions_split: DatasetSplit = fold.split_dataset(predictions)

    # TODO: change to list of dicts
    predictions_dict = {}
    for set_name, dataset in dataset_split.__dict__.items():
        y_target = dataset.y
        y_hat = getattr(predictions_split, set_name).y

        predictions_dict[set_name] = {
            "targets": y_target[~np.isnan(y_target)].tolist(),
            "predictions": y_hat[~np.isnan(y_hat)].tolist(),
        }

    return estimator, predictions_dict


def run_error_callback(exception: Exception):
    try:
        raise exception
    except KeyboardInterrupt:  # Should be unnecessary
        raise
    except RunException as run_exception:
        client = run_exception.run_executor.client
        run_id = run_exception.run_executor.run_id
        client.log_text(run_id, str(run_exception), "error_message.txt")
        client.log_text(run_id, traceback.format_exc(), "error_traceback.txt")
        client.set_terminated(run_id, status="FAILED")
    except:  # Should be unnecessary
        raise


# TODO: define error messages
# TODO: use mlflow Dataset API
# TODO: save models/predictions
# TODO: rename fold to split?


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
            # artifact_location=artifact_location,  # TODO
            description=experiment_data["description"],
        )
        experiment = client.get_experiment(experiment_id)
        if not os.access(uri_to_path(experiment.artifact_location), os.W_OK):
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
            folds = [
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
            ]
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
                    pool.apply_async(run_executor, error_callback=run_error_callback)
                )

    print("Running...")
    try:
        pool.close()
        for item in jobs:
            item.wait(timeout=9999999)  # Without a timeout, we can't interrupt this.
    except KeyboardInterrupt:
        pool.terminate()
    finally:
        pool.join()
        print("Finished.")


if __name__ == "__main__":
    main()
