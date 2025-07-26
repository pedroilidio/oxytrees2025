from abc import abstractmethod
import sys
import functools
from pathlib import Path
from dataclasses import dataclass, field, replace
import traceback
from typing import Any, Callable, Iterable, final
from warnings import warn
from importlib import import_module
import time

import numpy as np
import numpy.typing as npt
import mlflow.tracking
import mlflow.entities
from sklearn.base import BaseEstimator, clone

sys.path.append(str(Path(__file__).parent))  # HACK

from mlflow_utils import log_sklearn_model  # TODO: avoid


class RunException(RuntimeError):
    def __init__(self, original_exception: Exception, run_executor: "BaseRunExecutor"):
        self.original_exception = original_exception
        self.run_executor = run_executor

    def __str__(self):
        return f"RunException: {self.original_exception}"

    def __reduce__(self):  # For pickling support
        return (self.__class__, (self.original_exception, self.run_executor))


@dataclass(kw_only=True, frozen=True)
class PredictionRecord:
    name: str
    targets: npt.NDArray[np.float64]
    predictions: npt.NDArray[np.float64]
    step: int = 0  # Step in the training process, if applicable.

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "targets": self.targets.tolist(),
            "predictions": self.predictions.tolist(),
            "step": self.step,
        }


@dataclass(kw_only=True, frozen=True)
class Dataset:
    name: str
    X: list[npt.NDArray[np.float32]]
    y: npt.NDArray[np.float64]
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
    test_rows: npt.NDArray[int]
    test_cols: npt.NDArray[int]
    test_dyads: npt.NDArray[int] = field(default_factory=list)
    # Dyads that are considered missing are used both in the learning and test sets.
    # For positive-unlabeled learning, these are the positions of zeros in the LL set.
    missing_dyads: npt.NDArray[int] = field(default_factory=list)

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
            y=y[np.ix_(self.test_rows, self.test_cols)],
        )
        TL = replace(
            dataset,
            name=f"{dataset.name}_{self.name}_TL",
            X=[X_rows_test, X_cols_train],
            y=y[np.ix_(self.test_rows, train_cols)],
        )
        LT = replace(
            dataset,
            name=f"{dataset.name}_{self.name}_LT",
            X=[X_rows_train, X_cols_test],
            y=y[np.ix_(train_rows, self.test_cols)],
        )
        LL = replace(
            dataset,
            name=f"{dataset.name}_{self.name}_LL",
            X=[X_rows_train, X_cols_train],
            y=y[np.ix_(train_rows, train_cols)],
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


@dataclass(kw_only=True)
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
class BaseRunExecutor:
    client: mlflow.tracking.MlflowClient
    experiment_id: str
    estimator_loader: EstimatorLoader
    metric_function_loaders: list[PythonFunctionLoader]
    dataset_loader: DatasetLoader
    fold: Fold
    skip_finished: bool = True
    name_tag_order = ("estimator", "dataset", "fold_name")
    _run_id: str | None = field(init=False, default=None)

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

    @abstractmethod
    def apply_estimator(self) -> tuple[BaseEstimator, list[PredictionRecord]]:
        """Apply the estimator to the dataset and fold.

        Returns:
            A tuple of fitted estimator and a list of PredictionRecord.
        """
        raise NotImplementedError("Subclasses must implement apply_estimator method.")

    def log_metrics(self, predictions: Iterable[PredictionRecord]):
        """Compute and log metrics for the predictions."""
        self.check_run_active()
        run_id = self._run_id

        result_records = []

        for metric_function_loader in self.metric_function_loaders:
            metric_name = metric_function_loader.name
            metric_func, _ = metric_function_loader.load()

            for pred in predictions:
                setting_name = pred.name
                step = pred.step
                score_name = f"{setting_name}__{metric_name}"

                try:  # Ignore scoring errors
                    score = metric_func(pred.targets, pred.predictions)
                    result_records.append(
                        {
                            "metric": metric_name,
                            "setting_name": setting_name,
                            "value": score,
                            "step": step,
                            "score_name": score_name,
                            # "timestamp": mlflow.utils.time.get_current_time_millis(),
                            "timestamp": int(time.time() * 1000)
                        }
                    )

                except ValueError as e:
                    self.client.log_text(
                        run_id, str(e), f"{score_name}__error_message.txt"
                    )
                    self.client.log_text(
                        run_id,
                        traceback.format_exc(),
                        f"{score_name}__error_traceback.txt",
                    )

        self.client.log_batch(
            metrics=[
                mlflow.entities.Metric(
                    key=rec["score_name"],
                    value=rec["value"],
                    timestamp=rec["timestamp"],
                    step=rec["step"],
                )
                for rec in result_records
            ],
            run_id=str(run_id),
        )
        return result_records

    def create_run(self) -> mlflow.entities.Run | None:
        """Create a new run in MLflow. Return None if the run should be skipped."""
        if self.skip_finished:
            print(f"Checking if run {self.name} was already executed...")
            finished_runs = self.client.search_runs(
                filter_string=(
                    # f"run_name = '{self.name}' AND status = 'FINISHED'"
                    f"run_name = '{self.name}' AND status != 'FAILED'"
                ),
                run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
                experiment_ids=[self.experiment_id],
                max_results=1,
            )
            if finished_runs:
                warn(f"Skipping already finished run: {self.name}")
                return None

        return self.client.create_run(
            experiment_id=self.experiment_id,
            run_name=self.name,
            tags=self.tags,
        )

    def execute(self):
        """Execute the run, implementing the specific logic for the run."""
        self._mlflow_run = self.create_run()
        if self._mlflow_run is None:
            return

        run_id = self._run_id = self._mlflow_run.info.run_id

        self.client.log_dict(
            run_id, self.dataset_loader.to_json(), "dataset_loader.yml"
        )
        self.client.log_dict(run_id, self.fold.to_json(), "fold_definition.yml")

        # Load data, split it, train the estimator, and return predictions
        fitted_estimator, predictions = self.apply_estimator()

        # Log fitted estimator to the model registry
        log_sklearn_model(
            self.client,
            run_id,
            fitted_estimator,
            self.estimator_loader.object_path,
        )

        # Log predictions
        self.client.log_dict(
            run_id, [p.to_json() for p in predictions], "predictions.yml"
        )
        # Calculate and log scores
        self.log_metrics(predictions)

        self.client.set_terminated(run_id)

    def check_run_active(self):
        """Check if the run corresponds to an active MLflow run."""
        for attr in ("_mlflow_run", "_run_id"):
            if not getattr(self, attr, None):
                raise ValueError("Run ID is not set. Cannot check run status.")

    def _execute_and_wrap_errors(self):
        try:
            self.execute()
        except KeyboardInterrupt:
            self.client.set_terminated(self._run_id, status="FAILED")
            raise
        except Exception as e:
            self.client.set_terminated(self._run_id, status="FAILED")
            self.client.log_text(
                self._run_id, traceback.format_exc(), "error_traceback.txt"
            )
            raise RunException(original_exception=e, run_executor=self) from e

    __call__ = _execute_and_wrap_errors

    def error_callback(self, exception: BaseException):
        """Handle errors during the run execution.

        Will be called by the multiprocessing pool when an error occurs.
        It logs the error and sets the run status to FAILED.
        """
        try:
            raise exception
        except KeyboardInterrupt:
            raise
        except RunException as run_exception:
            warn("RunException occurred: " + str(run_exception))
