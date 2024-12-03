from importlib import import_module
import multiprocessing as mp
import os
from pathlib import Path
import sys

import yaml
import click
import sys
import numpy as np

rng = np.random.default_rng(0)

TEST_DATA = {
    "X": [rng.random((5, 5)), rng.random((7, 7))],
    "y": rng.integers(0, 2, size=(5, 7)).astype("float64"),
    "pairwise": True,
}


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# TODO: this is copied from run_experiments.py, move to helper scripts
def load_python_object(object_path: str, code_paths: list[Path] | None = None):
    module_name, obj_name = object_path.rsplit(".", 1)

    if code_paths:
        sys.path.extend(str(p.resolve()) for p in code_paths)

    module = import_module(module_name)

    if code_paths:
        sys.path = sys.path[: -len(code_paths)]

    return getattr(module, obj_name), module.__file__


def check_estimator(estimator_object_path, code_paths):
    try:
        estimator, _ = load_python_object(estimator_object_path, code_paths)
    except Exception as e:
        raise AssertionError(f"Error loading estimator: {e}")

    try:
        estimator.fit(TEST_DATA["X"], TEST_DATA["y"])
    except Exception as e:
        raise AssertionError(f"Error fitting estimator: {e}")

    try:
        pred = estimator.predict([TEST_DATA["X"][0][:3, :], TEST_DATA["X"][1][:2, :]])
        assert pred.shape == (6,), f"Unexpected prediction shape: {pred.shape}"
    except Exception as e:
        raise AssertionError(f"Error predicting with estimator: {e} != (35,)")


def check_estimators(estimators_path: Path, code_paths: list[Path], n_jobs: int):
    with estimators_path.open() as f:
        estimators = yaml.safe_load(f)

    count_failed = 0
    count_timedout = 0
    with mp.Pool(n_jobs) as pool:
        for estimator_name, estimator_path in estimators.items():
            try:
                proc = pool.apply_async(check_estimator, (estimator_path, code_paths))
                proc.get(timeout=30)
            except AssertionError as e:
                count_failed += 1
                print(f"* [X] {estimator_name}: {e}")
            except mp.TimeoutError as e:
                count_timedout += 1
                print(f"* [X] {estimator_name}: {e}")
            else:
                print(f"[OK!] {estimator_name}")

    print(
        f"\nChecks finished. {count_failed} estimators failed."
        f" {count_timedout} timed out."
    )


@click.command()
@click.option(
    "--estimator-definitions",
    "-d",
    type=click.Path(exists=True, path_type=Path, dir_okay=False),
    required=True,
    help="Path to the estimator definitions YAML file.",
)
@click.option(
    "--code-path",
    "-c",
    "-b",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    multiple=True,
    help="Path to directory from where unfitted estimators can be imported.",
)
@click.option(
    "--n-jobs",
    "-j",
    type=int,
    default=1,
    help="Number of parallel jobs to run.",
)
def main(estimator_definitions, code_path, n_jobs):
    """Check the estimators defined in the given YAML file."""
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[var] = "1"

    check_estimators(estimator_definitions, list(code_path), n_jobs)


if __name__ == "__main__":
    main()
