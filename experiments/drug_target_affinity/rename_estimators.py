import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def rename_estimators(input_path: Path):
    print(f"Renaming estimators from {input_path}...")
    data = pd.read_table(input_path)
    output_path = input_path.with_stem(input_path.stem + "_renamed")

    dataset_renaming = {
        "kiba_raw": "kiba",
        "davis_raw": "davis",
    }

    # Rename datasets
    data["dataset.name"] = (
        data["dataset.name"].map(dataset_renaming).fillna(data["dataset.name"])
    )
    data.to_csv(output_path, sep="\t", index=False)

    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Rename estimators in a runs table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "results",
        type=Path,
        help="Path to the runs table to rename.",
    )
    args = parser.parse_args()

    rename_estimators(args.results)


if __name__ == "__main__":
    main()
