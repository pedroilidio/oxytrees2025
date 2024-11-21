import traceback
import logging
from typing import Any
from types import ModuleType
import sys
import warnings
from pathlib import Path
from itertools import product
from importlib import import_module
import multiprocessing as mp

import numpy as np
import yaml
import click
import joblib
from sklearn.base import clone
import mlflow
from mlflow import MlflowClient
from mlflow.types import TensorSpec, Schema
from mlflow.models.signature import ModelSignature

# TODO: dependency injection for folds, dataset splits, etc.

# TODO: include number of features (the one bellow is used only for unfit models)
BIPARTITE_SIGNATURE = ModelSignature(
    inputs=Schema([TensorSpec(type=np.dtype("float32"), shape=(2, -1, -1))]),
    outputs=Schema([TensorSpec(type=np.dtype("float64"), shape=(-1,))]),
)


def fold_run_is_finished(*, client, experiment_id, fold_index, parent_run_id):
    runs = client.search_runs(
        filter_string=(
            f"tags.mlflow.parentRunId = '{parent_run_id}'"
            f" AND tags.fold_index = '{fold_index}'"
            f" AND status = 'FINISHED'"
        ),
        experiment_ids=experiment_id,
        max_results=1,
    )
    return bool(runs)  # If any run is found, the fold is considered finished


def get_run_id_from_name(*, client, experiment_id, run_name, tags):
    runs = client.search_runs(
        experiment_ids=experiment_id,
        filter_string=f"run_name = '{run_name}'",
        max_results=1,
    )
    if runs:
        run_id = runs[0].info.run_id
        client.update_run(run_id, status="RUNNING")
        return run_id

    return client.create_run(
        run_name=run_name,
        experiment_id=experiment_id,
        tags=tags,
    ).info.run_id


def get_experiment_id_from_name(
    *, client, experiment_name, artifact_location, description
):
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        return client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
            tags={"mlflow.note.content": description}
        )

    return experiment.experiment_id


# HACK
# FIXME: unnecessary modules are being listed
def iter_relative_module_paths(module):
    for name, imported_module in sys.modules.items():
        if not hasattr(imported_module, "__package__"):
            continue
        if imported_module.__package__ is None:
            continue
        if not imported_module.__package__.startswith(module.__package__):
            continue
        if imported_module.__file__ is None:
            continue

        yield imported_module.__file__


def log_sklearn_model(client, run_id, estimator, code_paths):
    # TODO
    # mlflow.models.Model.log(
    #     run_id=run_id,
    #     sk_model=estimator,
    #     artifact_path="model",
    #     flavor=mlflow.sklearn,
    #     signature=BIPARTITE_SIGNATURE,
    #     code_paths=code_paths,
    #     # TODO: Tensors are currently not supported by MLflow
    #     # input_example=[X[0][:2], X[1][:3]],
    # )
    str_params = {k: str(v) for k, v in estimator.get_params().items()}  # FIXME
    client.log_dict(run_id, str_params, "estimator_params.yml")
    client.log_param(run_id, "estimator_class", estimator.__class__.__name__)
    client.log_param(run_id, "estimator_module", code_paths[0])


def load_python_object(
    object_path: str,
    code_paths: list[Path] | None = None,
) -> tuple[Any, str]:

    module_name, obj_name = object_path.rsplit(".", 1)

    if code_paths:
        sys.path.extend(str(p.resolve()) for p in code_paths)

    module = import_module(module_name)
    # relative_modules = list(iter_relative_module_paths(module))  # TODO
    relative_modules = []

    if code_paths:
        sys.path = sys.path[: -len(code_paths)]

    return getattr(module, obj_name), [module.__file__, *relative_modules]


# Just for modularity, may change in the future
def load_matrix(path):
    return np.load(path)


def split_bipartite_dataset(X, y, *, test_rows, test_cols, pairwise):
    # We are being repetitive here, but it's better to be explicit.
    X_rows, X_cols = X
    train_rows = np.delete(np.arange(X_rows.shape[0]), test_rows)
    train_cols = np.delete(np.arange(X_cols.shape[0]), test_cols)

    X_rows_test = X_rows[test_rows, :]
    X_rows_train = X_rows[train_rows, :]

    X_cols_test = X_cols[test_cols, :]
    X_cols_train = X_cols[train_cols, :]

    if pairwise:
        X_rows_test = X_rows_test[:, train_rows]
        X_rows_train = X_rows_train[:, train_rows]

        X_cols_test = X_cols_test[:, train_cols]
        X_cols_train = X_cols_train[:, train_cols]

    return {
        "TT": (X_rows_test, X_cols_test, y[test_rows, :][:, test_cols]),
        "TL": (X_rows_test, X_cols_train, y[test_rows, :][:, train_cols]),
        "LT": (X_rows_train, X_cols_test, y[train_rows, :][:, test_cols]),
        "LL": (X_rows_train, X_cols_train, y[train_rows, :][:, train_cols]),
    }


def apply_estimator(estimator, masked_positives, dataset_split):
    X_rows_LL, X_cols_LL, y_LL = dataset_split["LL"]

    if masked_positives:
        y_LL_masked = y_LL.copy()
        y_LL_masked.flat[masked_positives] = 0
        masked_indices = np.where(y_LL_masked.flat == 0)
    else:
        y_LL_masked = y_LL

    estimator = clone(estimator)
    estimator.fit([X_rows_LL, X_cols_LL], y_LL_masked)

    predictions = {
        key: {
            "targets": y.reshape(-1).tolist(),
            "predictions": estimator.predict([X_rows, X_cols]).tolist(),
        }
        for key, (X_rows, X_cols, y) in dataset_split.items()
        if y.size > 0
    }

    if masked_positives:
        y_true = predictions["LL"]["targets"]
        y_pred = predictions["LL"]["predictions"]

        # TODO: avoid conversion to array again
        predictions["LL_M"] = {
            "targets": np.array(y_true)[masked_indices].tolist(),
            "predictions": np.array(y_pred)[masked_indices].tolist(),
        }

    return estimator, predictions


def execute_fold_run(
    *,
    client,
    estimator,
    scoring_functions,
    dataset_data,
    X,
    y,
    fold_index,
    fold_definition,
    parent_run_id,
    experiment_id,
):
    fold_run = client.create_run(
        experiment_id=experiment_id,
        run_name=f"fold_{fold_index}",
        tags={"fold_index": fold_index, "mlflow.parentRunId": parent_run_id},
    )
    fold_run_id = fold_run.info.run_id
    client.log_dict(fold_run.info.run_id, fold_definition, "fold_definition.yml")

    try:
        dataset_split = split_bipartite_dataset(
            X,
            y,
            test_rows=fold_definition["test_rows"],
            test_cols=fold_definition["test_cols"],
            pairwise=dataset_data["pairwise"],
        )
        estimator, predictions = apply_estimator(
            estimator=estimator,
            masked_positives=fold_definition["masked_positives"],
            dataset_split=dataset_split,
        )
        client.log_dict(fold_run_id, predictions, "predictions.yml")

        for scoring_name, scoring_func in scoring_functions.items():
            for setting_name, pred in predictions.items():
                client.log_metric(
                    fold_run_id,
                    f"{setting_name}__{scoring_name}",
                    scoring_func(pred["targets"], pred["predictions"]),
                )

        client.set_terminated(fold_run_id)

    except KeyboardInterrupt as e:
        client.log_text(fold_run_id, str(e), "error_message.txt")
        client.log_text(fold_run_id, traceback.format_exc(), "error_traceback.txt")
        client.set_terminated(fold_run_id, status="INTERRUPTED")
        raise

    except Exception as e:
        client.log_text(fold_run_id, str(e), "error_message.txt")
        client.log_text(fold_run_id, traceback.format_exc(), "error_traceback.txt")
        client.set_terminated(fold_run_id, status="FAILED")


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
    "--output-directory",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory for storing results.",
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
    output_directory,
    n_jobs,
    code_path,
    skip_finished=False,
):
    if n_jobs < 1:
        n_jobs = mp.cpu_count() + n_jobs

    sys.path.extend(map(str, code_path))  # HACK

    artifact_location = output_directory.resolve().as_uri()
    client = MlflowClient(tracking_uri=artifact_location)
    pool = mp.Pool(n_jobs)

    logger = logging.getLogger("mlflow")
    logger.setLevel(logging.DEBUG)

    estimators = yaml.safe_load(estimator_definitions)
    datasets = yaml.safe_load(dataset_definitions)
    fold_definitions = yaml.safe_load(fold_definitions)
    experiments = yaml.safe_load(experiment_definitions)
    scoring_definitions = yaml.safe_load(scoring_definitions)

    order = ("validation_setting", "estimator", "dataset")

    try:
        for experiment_name, experiment_data in experiments.items():
            if experiment_data["active"] is False:
                warnings.warn(f"Skipping inactive experiment_data: {experiment_name}")
                continue

            experiment_id = get_experiment_id_from_name(
                client=client,
                experiment_name=experiment_name,
                artifact_location=artifact_location,
                description=experiment_data["description"],
            )
            scoring_functions = {
                scoring_name: load_python_object(
                    scoring_definitions[scoring_name],
                    code_path,
                )[0]
                for scoring_name in experiment_data["scoring"]
            }

            for run_data in product(*(experiment_data[key] for key in order)):
                run_dict = {key: value for key, value in zip(order, run_data)}
                run_name = "{dataset}__{estimator}__{validation_setting}".format(
                    **run_dict
                )

                run_id = get_run_id_from_name(
                    client=client,
                    experiment_id=experiment_id,
                    run_name=run_name,
                    tags=run_dict,
                )

                dataset_data = datasets[run_dict["dataset"]]
                folds = fold_definitions[run_dict["dataset"]][
                    run_dict["validation_setting"]
                ]
                estimator, estimator_code_paths = load_python_object(
                    estimators[run_dict["estimator"]], code_path
                )

                X = [load_matrix(path) for path in dataset_data["X"]]
                y = load_matrix(dataset_data["y"])

                log_sklearn_model(client, run_id, estimator, estimator_code_paths)

                for fold_index, fold_definition in enumerate(folds):
                    if skip_finished:
                        if fold_run_is_finished(
                            client=client,
                            experiment_id=experiment_id,
                            parent_run_id=run_id,
                            fold_index=fold_index,
                        ):
                            warnings.warn(
                                f"Skipping finished run: {run_name},"
                                f" fold {fold_index}"
                            )
                            continue

                    pool.apply_async(
                        execute_fold_run,
                        kwds=dict(
                            client=client,
                            parent_run_id=run_id,
                            experiment_id=experiment_id,
                            estimator=estimator,
                            X=X,
                            y=y,
                            scoring_functions=scoring_functions,
                            dataset_data=dataset_data,
                            fold_index=fold_index,
                            fold_definition=fold_definition,
                        ),
                    )

                pool.apply_async(client.set_terminated, args=(run_id,))

        pool.close()
        pool.join()

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        raise


if __name__ == "__main__":
    main()
