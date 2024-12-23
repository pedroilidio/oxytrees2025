import gc
from time import process_time_ns
import traceback
from typing import Any
import os
import sys
from pathlib import Path
from importlib import import_module
from itertools import count
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
from bipartite_learn.tree import BipartiteExtraTreeRegressor

EXPERIMENT_NAME = "empirical_complexity_predict"
EXPERIMENT_DESCRIPTION = "Empirical complexity of predict function of bipartite trees."
START_SIZE = 50
FACTOR = 1.01
RANDOM_STATE = 0
LOCK = mp.Lock()

ESTIMATOR = BipartiteExtraTreeRegressor(
    criterion="squared_error_gso",
    max_depth=None,
    max_features=1.0,
    random_state=0,
)

# TODO: include number of features (the one bellow is used only for unfit models)
BIPARTITE_SIGNATURE = ModelSignature(
    inputs=Schema([TensorSpec(type=np.dtype("float32"), shape=(2, -1, -1))]),
    outputs=Schema([TensorSpec(type=np.dtype("float64"), shape=(-1,))]),
)


def error_callback(e):
    print(e)
    if isinstance(e, KeyboardInterrupt):
        raise e


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


def log_sklearn_model(client, run_id, estimator):
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


def set_n_estimators(estimator, n_estimators):
    try:
        return estimator.set_params(n_estimators=n_estimators)
    except ValueError:
        return estimator.estimator.set_params(n_estimators=n_estimators)


def execute_run(
    *,
    client,
    experiment_id,
    estimator,
    tags,
):
    # TODO: dependency injection to tags
    iteration = tags["iteration"]
    data_size = tags["data_size"]
    random_state = tags["random_state"]

    order = ("data_size", "iteration", "random_state")
    run = client.create_run(
        experiment_id=experiment_id,
        run_name="__".join(str(tags[k]) for k in order),
        tags=tags,
    )
    run_id = run.info.run_id

    try:
        rng = np.random.default_rng(random_state + iteration)
        X = [
            np.asfortranarray(rng.random((data_size, data_size), dtype=np.float32))
            for _ in range(2)
        ]
        y = np.ascontiguousarray(rng.random((data_size, data_size), dtype=np.float64))

        estimator = clone(estimator)
        # TODO: Log fitted model?
        log_sklearn_model(client, run_id, estimator)

        start = process_time_ns()
        estimator.fit(X, y)
        fit_time = process_time_ns() - start

        client.log_metric(run_id, "fit_time", fit_time, step=iteration)

        tree = estimator.tree_

        start = process_time_ns()
        tree.apply(X)
        predict_time = process_time_ns() - start

        client.log_metric(run_id, "predict_time", predict_time, step=iteration)

        LOCK.acquire()
        try:
            molten_X = np.asfortranarray(row_cartesian_product(X))

            start = process_time_ns()
            tree.apply(molten_X)
            predict_time_single = process_time_ns() - start

            del molten_X
            gc.collect()
        finally:
            LOCK.release()

        client.log_metric(
            run_id, "predict_time_single", predict_time_single, step=iteration
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
    "--skip-finished",
    is_flag=True,
    help="Skip experiments that have already been run.",
)
def main(
    tracking_uri,
    n_jobs,
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

    client = MlflowClient(tracking_uri=tracking_uri)

    experiment_id = get_experiment_id_from_name(
        client=client,
        experiment_name=EXPERIMENT_NAME,
        # artifact_location=artifact_location,  # TODO
        description=EXPERIMENT_DESCRIPTION,
    )
    # Define the order of the tags to be used for checking if a run already exists.
    order = ("data_size", "iteration", "random_state")

    if skip_finished:
        print("Collecting finished runs...")
        finished_runs = client.search_runs(
            filter_string="status = 'FINISHED'",
            experiment_ids=experiment_id,
            max_results=50_000,  # Maximum allowed by MLflow
        )
        finished_runs = {
            tuple(str(run.data.tags[k]) for k in order)
            for run in tqdm(finished_runs, desc="Processing finished runs")
        }

    current = START_SIZE
    factor = FACTOR

    pool = mp.Pool(n_jobs)

    try:
        for iteration in count():
            data_size = int(current)
            current *= factor

            # Get the size of the molten X matrix
            # 4 bytes for each float32 element
            molten_size = (data_size**2) * (data_size * 2) * 4

            # Stop if the molten matrix is larger than 100 GB
            if molten_size > 100e9:
                print(
                    f"Stopping at data_size={data_size}"
                    f" (molten_size={molten_size/1e9:.2f} GB)"
                )
                break

            tags = {
                "data_size": data_size,
                "iteration": iteration,
                "random_state": RANDOM_STATE,
            }
            if skip_finished and tuple(str(tags[k]) for k in order) in finished_runs:
                print(f"Skipping {tags}")
                continue

            print(f"Queuing {tags}")

            pool.apply_async(
                execute_run,
                kwds=dict(
                    client=client,
                    experiment_id=experiment_id,
                    estimator=ESTIMATOR,
                    tags=tags,
                ),
                error_callback=error_callback,
            )

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
