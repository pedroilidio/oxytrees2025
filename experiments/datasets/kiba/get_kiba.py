import argparse
import io
import requests
import zipfile
from pathlib import Path

import pandas as pd

from compute_similarities import (
    compute_compound_similarity_matrix,
    compute_protein_similarity_matrix,
    normalize_similarity_matrix,
)

# Original sources:
# https://pubs.acs.org/doi/10.1021/ci400709d
# https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0209-z
KIBA_URL = "https://github.com/futianfan/DeepPurpose_Data/raw/main/KIBA.zip"


def build_similarity_matrices(
    *,
    ligand_smiles: pd.Series,
    target_seqs: pd.Series,
    ligand_similarity_output: Path = Path("ligand_similarity.tsv"),
    target_similarity_output: Path = Path("target_similarity.tsv"),
    norm_target_output: Path = Path("norm_target_similarity.tsv"),
    n_jobs: int = 1,
) -> pd.DataFrame:
    print("Calculating ligand-ligand similarities...")
    ligand_similarity_matrix = compute_compound_similarity_matrix(
        ligand_smiles,
        n_jobs=n_jobs,
    )
    ligand_similarity_output.parent.mkdir(parents=True, exist_ok=True)
    ligand_similarity_matrix.to_csv(ligand_similarity_output, sep='\t')
    print(f'ligand similarity matrix saved to {ligand_similarity_output}.')

    print("Calculating target-target similarities...")
    target_similarity_matrix = compute_protein_similarity_matrix(
        target_seqs,
        n_jobs=n_jobs,
    )
    target_similarity_output.parent.mkdir(parents=True, exist_ok=True)
    target_similarity_matrix.to_csv(target_similarity_output, sep='\t')
    print(f'Target protein similarity matrix saved to {target_similarity_output}.')

    norm_target_similarity_matrix = normalize_similarity_matrix(target_similarity_matrix)
    norm_target_output.parent.mkdir(parents=True, exist_ok=True)
    norm_target_similarity_matrix.to_csv(norm_target_output, sep='\t')
    print(f'Normalized target protein similarity matrix saved to {norm_target_output}.')


def download_kiba(path_kiba: Path = Path("KIBA")) -> Path:
    print(f"KIBA dataset will be saved to '{path_kiba.resolve()}'.")

    if path_kiba.exists():
        print("KIBA already downloaded.")
        return path_kiba

    print("Downloading and unzipping KIBA...", end=" ")
    with requests.get(KIBA_URL, stream=True) as response:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            zip_file.extractall(
                members=[
                    'KIBA/',
                    'KIBA/affinity.txt',
                    'KIBA/SMILES.txt',
                    'KIBA/target_seq.txt'
                ]
            )
    print("Done!")
    return path_kiba


def parse_kiba(
    kiba_dir: Path,
    final_dir: Path = Path("final"),
    n_jobs: int = 1,
) -> Path:
    # Load SMILES json
    smiles = (
        pd.read_json(kiba_dir / "SMILES.txt", typ="series")
        .rename_axis(index="compound_id")
        .rename("smiles")
    )
    # Load target sequences json
    target_seq = (
        pd.read_json(kiba_dir / "target_seq.txt", typ="series")
        .rename_axis(index="target_id")
        .rename("sequence")
    )

    # Load and binarize affinity matrix
    affinity = (
        pd.read_table(kiba_dir / "affinity.txt", names=target_seq.index)
        .set_index(smiles.index)
    )

    # The dataset was preprocessed by DOI:10.1186/s13321-017-0209-z to remove
    # compounds and targets with less than 10 interactions. The values were also 
    # made non-negative by subtracting the minimum value (which was negative).
    # The threshold for binarization is thus 12.1 according to the same authors.
    binary_affinity = (affinity <= 12.1).astype(int)

    final_dir.mkdir(parents=True, exist_ok=True)
    smiles.to_csv(final_dir / "smiles.tsv", sep="\t")
    target_seq.to_csv(final_dir / "target_sequences.tsv", sep="\t")
    affinity.to_csv(final_dir / "affinity.tsv", sep="\t")
    binary_affinity.to_csv(final_dir / "binary_affinity.tsv", sep="\t")
    print(f"Formatted tables saved to {final_dir.resolve()}.")

    print("Bulding similarity matrices...")
    build_similarity_matrices(
        ligand_smiles=smiles,
        target_seqs=target_seq,
        ligand_similarity_output=final_dir / "ligand_similarity.tsv",
        target_similarity_output=final_dir / "target_similarity.tsv",
        norm_target_output=final_dir / "normalized_target_similarity.tsv",
        n_jobs=n_jobs,
    )

    print(f"Processed KIBA dataset saved to '{final_dir.resolve()}'.")
    return final_dir


def main() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--final-dir",
        "--final_dir",
        type=Path,
        default=Path("final"),
        help="Directory where the final dataset will be saved.",
    )
    argparser.add_argument(
        "--n-jobs",
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs to use.",
    )
    args = argparser.parse_args()

    # Download and unzip KIBA
    kiba_dir = download_kiba()

    # Format KIBA to tables and save to final_dir
    parse_kiba(
        kiba_dir=kiba_dir,
        final_dir=args.final_dir,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()