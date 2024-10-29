from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import yaml
from tqdm import tqdm
from run_experiments import load_dataset, deep_update


def collect_dataset_metadata(config_file, output_file):
    print(f"Loading datasets from {config_file}.")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    datasets_load_info = deep_update(
        config["defaults"]["aliases"]["dataset"],
        config["aliases"]["dataset"],
    )
    datasets = tqdm(
        map(load_dataset, datasets_load_info),
        total=len(datasets_load_info),
    )
    df = pd.DataFrame.from_records(
        (
            {
                "dataset_name": dataset["name"],
                "n_row_samples": dataset["y"].shape[0],
                "n_col_samples": dataset["y"].shape[1],
                "n_row_features": dataset["X"][0].shape[1],
                "n_col_features": dataset["X"][1].shape[1],
                "n_interactions": dataset["y"].size,
                "n_positive_interactions": int(dataset["y"].sum()),
                "density": dataset["y"].mean(),
            }
            for dataset in datasets
        )
    )
    df.to_csv(output_file, sep="\t", index=False)
    print(f"Saved dataset metadata to {output_file}.")


def main():
    argparser = ArgumentParser()
    argparser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config.yml"),
        help="Path to run_experiments.py's config file.",
    )
    argparser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("dataset_info.tsv"),
        help="Path to tabular output file.",
    )
    args = argparser.parse_args()
    collect_dataset_metadata(args.config, args.output)


if __name__ == "__main__":
    main()
