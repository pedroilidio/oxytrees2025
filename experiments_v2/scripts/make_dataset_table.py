from pathlib import Path

import pandas as pd
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
    X = [np.load(base_dir / path) for path in X_paths]
    y = np.load(base_dir / y_path)

    return X, y


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
@click.option(
    "--renaming",
    "-r",
    required=False,
    type=click.Path(exists=True, path_type=Path, dir_okay=False),
    help="Path to the YAML file containing name substitutions.",
)
def main(dataset_definitions: Path, base_dir: Path = Path.cwd(), renaming: Path = None):
    """Check the datasets defined in the given YAML file."""
    base_dir = base_dir.resolve()
    datasets_path = dataset_definitions.resolve()

    datasets = yaml.safe_load(datasets_path.read_text())
    renaming = renaming and yaml.safe_load(renaming.read_text())

    rows = []
    for dataset_name, dataset_info in datasets.items():
        # Load matrices
        X, y = load_dataset(dataset_info, base_dir)
        rows.append(
            {
                "Name": dataset_name,
                "Shape": "${} \\times {}$".format(*y.shape),
                "Density": f"{y.mean():.2g}",
            }
        )

    table = pd.DataFrame.from_records(rows)
    print(table.to_latex(index=False))


if __name__ == "__main__":
    main()
