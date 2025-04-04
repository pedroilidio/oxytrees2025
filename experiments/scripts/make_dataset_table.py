from pathlib import Path

import pandas as pd
import numpy as np
import yaml
import click

DATASET_DESCRIPTIONS = {
    "dpi_n": "Drug-Nuclear receptor",
    "dpi_g": "Drug-GPCR",
    "dpi_i": "Drug-Ion channel",
    "dpi_e": "Drug-Enzyme",
    "srn": "Gene-Transcription factor",
    "ern": "Gene-Transcription factor",
    "davis": "Inhibitor-Kinase",
    "kiba": "Inhibitor-Kinase",
    "mirtarbase": "miRNA-Target gene",
    "npinter": "lncRNA-Protein",
    "lncrna_cancer": "lncRNA-Cancer",
    "lncrna_disease": "lncRNA-Disease",
    "lncrna_mirna": "lncRNA-miRNA",
    "mirna_disease": "miRNA-Disease",
    "te_pirna": "Transposable element-piRNA",
}


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
    "--cv-config",
    "-c",
    type=click.Path(exists=True, path_type=Path, dir_okay=False),
    required=True,
    help="Path to the cross-validation configuration YAML file.",
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
def main(
    dataset_definitions: Path,
    cv_config: Path,
    base_dir: Path = Path.cwd(),
    renaming: Path = None,
):
    """Check the datasets defined in the given YAML file."""
    base_dir = base_dir.resolve()
    datasets_path = dataset_definitions.resolve()

    datasets = yaml.safe_load(datasets_path.read_text())
    if cv_config:
        cv_config = yaml.safe_load(cv_config.read_text())

    renaming = renaming and yaml.safe_load(renaming.read_text())["dataset"]
    renaming = renaming or {}

    rows = []
    for dataset_name, dataset_info in datasets.items():
        # Load matrices
        X, y = load_dataset(dataset_info, base_dir)
        n_folds = cv_config[dataset_name]["TT"]
        rows.append(
            {
                "Name": renaming.get(dataset_name, dataset_name),
                "Domain": DATASET_DESCRIPTIONS.get(dataset_name, ""),
                "Dimensionality": "${} \\times {}$".format(*y.shape),
                "Folds": f"${n_folds} \\times {n_folds}$",
                "Dyads": y.size,
                "Positives": int(y.sum()),
                "Density": f"{y.mean() * 100:0.3g}\\%",
            }
        )

    table = pd.DataFrame.from_records(rows).sort_values("# of dyads")
    print(table.to_latex(index=False))


if __name__ == "__main__":
    main()
