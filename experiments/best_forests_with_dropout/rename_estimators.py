import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def rename_estimators(input_path: Path):
    print(f"Renaming estimators from {input_path}...")
    data = pd.read_table(input_path)
    output_path = input_path.with_stem(input_path.stem + "_renamed")

    wrapper_renaming = {
        np.nan: "",
        "regressor_to_classifier": "",
        "nrlmf_y_reconstruction": "__nrlmf",
        "nrlmf_y_reconstruction_drop50": "__nrlmf__50",
        "nrlmf_y_reconstruction_drop70": "__nrlmf__70",
        "nrlmf_y_reconstruction_drop90": "__nrlmf__90",
        "drop50": "__50",
        "drop70": "__70",
        "drop90": "__90",
        "dnilmf_y_reconstruction": "__dnilmf",
    }
    estimator_renaming = {
        "bxt_gmo__25": "bxt_gmo",
        "bxt_gmo__75": "bxt_gmo",
        "brf_gmo__75": "brf_gmo",
        "ss_bxt_gso__md_size": "md_size",
        "ss_bxt_gso__md_fixed": "md_fixed",
        "ss_bxt_gso__ad_size": "ad_size",
        "ss_bxt_gso__ad_fixed": "ad_fixed",
        "ss_bxt_gso__mse_density": "mse_density",
    }

    # Rename estimators
    data["estimator.name"] = (
        data["estimator.name"].map(estimator_renaming).fillna(data["estimator.name"])
    )

    # Rename estimators to include wrapper name
    data["suffix"] = (
        data["wrapper.name"].map(wrapper_renaming).fillna(data["wrapper.name"])
    )
    data["estimator.name"] += data["suffix"]
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
