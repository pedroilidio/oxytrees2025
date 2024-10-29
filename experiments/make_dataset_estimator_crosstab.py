import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def make_dataset_estimator_crosstab(input_path: Path):
    print(f"Making dataset estimator crosstab from {input_path}...")
    output_path = input_path.with_stem(input_path.stem + "_crosstab")

    data = pd.read_table(input_path).sort_values('estimator.name')
    # TODO: subset as arg
    data = data.dropna(subset=["results.TT_average_precision", "results.TT_roc_auc"])

    crosstab = pd.crosstab(
        data["estimator.name"],
        data["dataset.name"],
    )

    # crosstab.to_csv(output_path, sep="\t")
    plt.figure(figsize=(10, 30))
    sns.heatmap(crosstab, annot=True, fmt="d", cbar=False, square=True)
    plt.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
    print(f"Saved to {output_path.with_suffix('.png')}")


def main():
    parser = argparse.ArgumentParser(
        description="Make a dataset estimator crosstab from a runs table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "results",
        type=Path,
        help="Path to the runs table to make the crosstab from.",
    )
    args = parser.parse_args()

    make_dataset_estimator_crosstab(args.results)


if __name__ == "__main__":
    main()
