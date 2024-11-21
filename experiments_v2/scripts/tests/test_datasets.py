from pathlib import Path

import numpy as np
import yaml
import click


def load_dataset(
    dataset_info: dict, base_dir: Path
) -> tuple[list[np.ndarray], np.ndarray, bool]:

    X_paths = dataset_info["X"]
    y_path = dataset_info["y"]
    pairwise = dataset_info["pairwise"]

    # Load matrices
    try:
        X = [np.load(base_dir / path) for path in X_paths]
        y = np.load(base_dir / y_path)
    except Exception as e:
        raise AssertionError(f"Error loading matrices: {e}")

    return X, y, pairwise


def check_dataset(dataset_info: dict, base_dir: Path = Path.cwd()):
    # Load matrices
    X, y, pairwise = load_dataset(dataset_info, base_dir)

    # Check shapes
    for i, Xi in enumerate(X):
        assert Xi.shape[0] == y.shape[i], (
            f"X[{i}] and y shapes do not match ({Xi.shape[0]} != {y.shape[i]})"
        )

    if pairwise:
        for i, Xi in enumerate(X):
            assert Xi.shape[0] == Xi.shape[1], (
                f"pairwise is set to True but X[{i}] matrix is not"
                f" square (shape: {X[i].shape})"
            )

    # Check y is binary
    y_values = np.unique(y)
    msg = f"y contains non-binary values: {y_values}"

    assert y_values.size == 2, msg
    np.sort(y_values)
    assert np.all(y_values == np.array([0, 1])), msg


def check_datasets(datasets_path: Path, base_dir: Path):
    base_dir = base_dir.resolve()
    datasets_path = datasets_path.resolve()

    with datasets_path.open() as f:
        datasets = yaml.safe_load(f)

    for dataset_name, dataset_info in datasets.items():
        try:
            check_dataset(dataset_info, base_dir)
        except AssertionError as e:
            print(f"* [X] {dataset_name}:", e)
        else:
            print(f"[OK!] {dataset_name}")

    print("\nChecks finished.")


@click.command()
@click.option(
    "--dataset-definitions",
    "-d",
    type=click.Path(exists=True, path_type=Path, dir_okay=False),
    required=True,
    help="Path to the dataset definitions YAML file.",
)
@click.option(
    "--base-dir",
    "-b",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
    help=(
        "Path to the base directory to which the paths in the dataset definitions are"
        " relative."
    ),
)
def main(dataset_definitions: Path, base_dir: Path = Path.cwd()):
    """Check the datasets defined in the given YAML file."""
    check_datasets(dataset_definitions, base_dir)


if __name__ == "__main__":
    main()
