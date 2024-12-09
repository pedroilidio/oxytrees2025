from time import process_time_ns
import traceback
from typing import Any
import os
import sys
import warnings
from pathlib import Path
from itertools import count
from importlib import import_module
import multiprocessing as mp

import numpy as np
import yaml
import click
from sklearn.base import clone
from mlflow import MlflowClient
from mlflow.types import TensorSpec, Schema
from mlflow.models.signature import ModelSignature
from tqdm import tqdm
from bipartite_learn.melter import row_cartesian_product

# TODO: dependency injection for folds, dataset splits, etc.

# TODO: include number of features (the one bellow is used only for unfit models)
BIPARTITE_SIGNATURE = ModelSignature(
    inputs=Schema([TensorSpec(type=np.dtype("float32"), shape=(2, -1, -1))]),
    outputs=Schema([TensorSpec(type=np.dtype("float64"), shape=(-1,))]),
)
EXPERIMENT_NAME = "empirical_complexity"
EXPERIMENT_DESCRIPTION = "Empirical complexity of bipartite forests."
CONFIG = {
    "estimators": [
        "bxt_bgso",
        "bxt_gmo",
        "uniform_bxt_bgso",
        "bxt_bgso_kronrls",
        "dwnn_similarities_bxt_bgso",
        "bxt_bgso_logistic",
    ],
    "random_state": 0,
    "factor": 1.1,
    "start": 10,
}


def get_experiment_id_from_name(*, client, experiment_name, description):
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        return client.create_experiment(
            name=experiment_name,
            # artifact_location=artifact_location,
            tags={"mlflow.note.content": description},
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


def execute_run(
    *,
    client,
    estimator,
    estimator_code_paths,
    X,
    y,
    experiment_id,
    iteration,
    estimator_name,
):
    data_size = y.shape[0]

    run = client.create_run(
        experiment_id=experiment_id,
        run_name="__".join((estimator_name, str(data_size), str(iteration))),
        tags={
            "iteration": iteration,
            "estimator": estimator_name,
            "data_size": data_size,
        },
    )
    run_id = run.info.run_id

    try:
        # TODO: Log fitted model?
        log_sklearn_model(client, run_id, estimator, estimator_code_paths)
        estimator = clone(estimator)
        start = process_time_ns()
        estimator.fit(X, y)
        fit_time = process_time_ns() - start

        client.log_metric(run_id, "fit_time", fit_time, step=iteration)

        start = process_time_ns()
        estimator.predict(X)
        predict_time = process_time_ns() - start

        client.log_metric(run_id, "predict_time", predict_time, step=iteration)

        try:
            start = process_time_ns()
            estimator.predict(row_cartesian_product(X))
            predict_time_bip = process_time_ns() - start
        except Exception as e:
            predict_time_bip = 0

        client.log_metric(
            run_id, "predict_time_bipartite", predict_time_bip, step=iteration
        )

        client.set_terminated(run_id)

    except Exception as e:
        client.log_text(run_id, str(e), "error_message.txt")
        client.log_text(run_id, traceback.format_exc(), "error_traceback.txt")
        client.set_terminated(run_id, status="FAILED")
        try:
            raise e
        except KeyboardInterrupt:
            client.update_run(run_id, status="INTERRUPTED")
            raise


@click.command()
@click.option(
    "--estimator-definitions",
    type=click.File("r"),
    required=True,
    help="YAML file with estimator definitions.",
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
    tracking_uri,
    n_jobs,
    code_path,
    skip_finished=False,
):
    if n_jobs < 1:
        n_jobs = mp.cpu_count() + n_jobs

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

    experiment_id = get_experiment_id_from_name(
        client=client,
        experiment_name=EXPERIMENT_NAME,
        # artifact_location=artifact_location,  # TODO
        description=EXPERIMENT_DESCRIPTION,
    )
    if skip_finished:
        print("Collecting finished runs...")
        finished_runs = client.search_runs(
            filter_string="status = 'FINISHED'",
            experiment_ids=[e.experiment_id for e in client.search_experiments()],
            max_results=50_000,  # Maximum allowed by MLflow
        )
        finished_runs = {
            tuple(map(run.data.tags.get, ("estimator", "data_size", "iteration")))
            for run in tqdm(finished_runs, desc="Processing finished runs")
        }

    current = CONFIG["start"]
    factor = CONFIG["factor"]
    rng = np.random.default_rng(CONFIG["random_state"])

    for iteration in count():
        data_size = int(current)
        current *= factor

        X1, X2, y = rng.random((3, data_size, data_size))

        for estimator_name in CONFIG["estimators"]:
            estimator_path = estimators[estimator_name]
            estimator, estimator_code_paths = load_python_object(
                estimator_path, code_path
            )

            pool.apply_async(
                execute_run,
                kwds=dict(
                    client=client,
                    experiment_id=experiment_id,
                    estimator=estimator,
                    estimator_code_paths=estimator_code_paths,
                    X=[X1, X2],
                    y=y,
                    iteration=iteration,
                    estimator_name=estimator_name,
                ),
            )

    # TODO: check if try here is correct
    try:
        print("Running...")
        pool.close()
        pool.join()

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        raise

    print("Done.")


if __name__ == "__main__":
    main()
