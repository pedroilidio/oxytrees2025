import warnings
from pathlib import Path
from itertools import product
from importlib import import_module

import numpy as np
import yaml
import click
import joblib
from sklearn.base import clone
import mlflow
from mlflow.types import TensorSpec, Schema
from mlflow.models.signature import ModelSignature

# TODO: rename masked_positives
# TODO: dependency injection for folds, dataset splits, etc.

BIPARTITE_SIGNATURE = ModelSignature(
    inputs=Schema([TensorSpec(type=np.dtype("float32"), shape=(2, -1, -1))]),
    outputs=Schema([TensorSpec(type=np.dtype("float64"), shape=(-1,))]),
)


def load_python_object(obj_path):
    module_name, obj_name = obj_path.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, obj_name)


# Just for modularity, may change in the future
def load_matrix(path):
    return np.loadtxt(path)


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
        key: (y.reshape(-1), estimator.predict([X_rows, X_cols]))
        for key, (X_rows, X_cols, y) in dataset_split.items()
        if y.size > 0
    }

    if masked_positives:
        y_LL, pred_LL = predictions["LL"]
        predictions["LL_M"] = (
            y_LL.flat[masked_indices],
            pred_LL.flat[masked_indices],
        )

    return estimator, predictions


def execute_fold_run(*, estimator, scoring_functions, dataset, X, y, fold_index, fold):
    with mlflow.start_run(run_name=f"fold_{fold_index}", nested=True):
        mlflow.set_tag("fold_index", fold_index)
        mlflow.log_dict(fold, "fold_definition.yml")

        dataset_split = split_bipartite_dataset(
            X,
            y,
            test_rows=fold["test_rows"],
            test_cols=fold["test_cols"],
            pairwise=dataset["pairwise"],
        )
        estimator, predictions = apply_estimator(
            estimator=estimator,
            masked_positives=fold["masked_positives"],
            dataset_split=dataset_split,
        )

        for scoring_name, scoring_func in scoring_functions.items():
            for setting_name, (y_true, y_pred) in predictions.items():
                mlflow.log_metric(
                    f"{scoring_name}__{setting_name}",
                    scoring_func(y_true, y_pred),
                )


# TODO: define error messages
# TODO: use mlflow Dataset API
# TODO: save models/predictions
# TODO: rename fold to split


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
    "--output-directory",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory for storing results.",
)
def main(
    estimator_definitions,
    dataset_definitions,
    fold_definitions,
    experiment_definitions,
    scoring_definitions,
    output_directory,
):
    mlflow.set_tracking_uri(output_directory.resolve().as_uri())
    # HACK: enable loading models from the same repository
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

    estimators = yaml.safe_load(estimator_definitions)
    datasets = yaml.safe_load(dataset_definitions)
    fold_definitions = yaml.safe_load(fold_definitions)
    experiments = yaml.safe_load(experiment_definitions)
    scoring_definitions = yaml.safe_load(scoring_definitions)

    order = ("validation_setting", "estimator", "dataset")

    for experiment_name, experiment in experiments.items():
        if experiment["active"] is False:
            warnings.warn(f"Skipping inactive experiment: {experiment_name}")
            continue
        mlflow.set_experiment(experiment_name=experiment_name)
        mlflow.set_experiment_tag("mlflow.note.content", experiment["description"])

        scoring_functions = {
            scoring_name: load_python_object(scoring_definitions[scoring_name])
            for scoring_name in experiment["scoring"]
        }

        for run_data in product(*(experiment[key] for key in order)):
            run_dict = {key: value for key, value in zip(order, run_data)}
            run_name = "{dataset}__{estimator}__{validation_setting}".format(**run_dict)

            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags(run_dict)

                dataset = datasets[run_dict["dataset"]]
                folds = fold_definitions[run_dict["dataset"]][
                    run_dict["validation_setting"]
                ]
                estimator = load_python_object(estimators[run_dict["estimator"]])

                X = [load_matrix(path) for path in dataset["X"]]
                y = load_matrix(dataset["y"])

                mlflow.sklearn.log_model(
                    estimator,
                    "estimator",
                    signature=BIPARTITE_SIGNATURE,
                    # TODO: Tensors are currently not supported by MLflow
                    # input_example=[X[0][:2], X[1][:3]],
                )

                joblib.Parallel(n_jobs=experiment["n_jobs"])(
                    joblib.delayed(execute_fold_run)(
                        estimator=estimator,
                        X=X,
                        y=y,
                        scoring_functions=scoring_functions,
                        dataset=dataset,
                        fold_index=fold_index,
                        fold=fold,
                    )
                    for fold_index, fold in enumerate(folds)
                )


if __name__ == "__main__":
    main()
