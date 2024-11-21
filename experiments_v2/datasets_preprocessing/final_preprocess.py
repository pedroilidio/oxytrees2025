from pathlib import Path
import warnings

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import click


def symmetrize(X):
    diff = np.abs(X - X.T)
    asymmetry_mask = diff > 1e-6

    if np.any(asymmetry_mask):
        warnings.warn(
            "Asymmetries detected in the input matrix. Absolute differences:"
            f"\n{diff[asymmetry_mask]}"
        )
    return (X + X.T) / 2


def transfer_dpi(input_dir: Path, output_dir: Path):
    X1 = np.loadtxt(input_dir / "X1.txt").astype("float32")
    X2 = np.loadtxt(input_dir / "X2.txt").astype("float32")
    y = np.loadtxt(input_dir / "Y.txt").astype("uint8")

    X1 = symmetrize(X1)
    X2 = symmetrize(X2)

    output_dir.mkdir()

    np.save(output_dir / "X1", X1)
    np.save(output_dir / "X2", X2)
    np.save(output_dir / "y", y)


def transfer_nuclear_receptors(input_dir: Path, output_dir: Path):
    transfer_dpi(input_dir / "nuclear_receptors", output_dir / "dpi_n")
    print("Transferred dpi_n")


def transfer_gpcr(input_dir: Path, output_dir: Path):
    transfer_dpi(input_dir / "gpcr", output_dir / "dpi_g")
    print("Transferred dpi_g")


def transfer_ion_channels(input_dir: Path, output_dir: Path):
    transfer_dpi(input_dir / "ion_channels", output_dir / "dpi_i")
    print("Transferred dpi_i")


def transfer_enzymes(input_dir: Path, output_dir: Path):
    transfer_dpi(input_dir / "enzymes", output_dir / "dpi_e")
    print("Transferred dpi_e")


def transfer_ern(input_dir: Path, output_dir: Path):
    X1 = np.loadtxt(input_dir / "ern/X1.txt", delimiter=",").astype("float32")
    X2 = np.loadtxt(input_dir / "ern/X2.txt", delimiter=",").astype("float32")
    y = np.loadtxt(input_dir / "ern/Y.txt", delimiter=",").astype("uint8")

    X1 = cosine_similarity(X1)
    X2 = cosine_similarity(X2)

    X1 = symmetrize(X1)
    X2 = symmetrize(X2)

    output_dir /= "ern"
    output_dir.mkdir()

    np.save(output_dir / "X1", X1)
    np.save(output_dir / "X2", X2)
    np.save(output_dir / "y", y)

    print("Transferred ern")


def transfer_srn(input_dir: Path, output_dir: Path):
    with (input_dir / "srn/var_names_1.txt").open() as f:
        X1_names = pd.Series(map(str.strip, f))
    X1 = (
        pd.read_csv(input_dir / "srn/X1.txt", header=None)
        .loc[:, ~X1_names.duplicated()]
        .set_axis(X1_names.drop_duplicates(), axis="columns")
        .loc[:, :"gene_number"]
        .drop(columns=["gene_number"])
        .values.astype("float32")
    )
    X2 = np.loadtxt(input_dir / "srn/X2.txt", delimiter=",").astype("float32")
    y = np.loadtxt(input_dir / "srn/Y.txt", delimiter=",").astype("uint8")

    X1 = cosine_similarity(X1)
    X2 = cosine_similarity(X2)

    X1 = symmetrize(X1)
    X2 = symmetrize(X2)

    output_dir /= "srn"
    output_dir.mkdir()

    np.save(output_dir / "X1", X1)
    np.save(output_dir / "X2", X2)
    np.save(output_dir / "y", y)

    print("Transferred srn")


def transfer_davis(input_dir: Path, output_dir: Path):
    X1 = np.loadtxt(input_dir / "davis/binary/X1.txt").astype("float32")
    X2 = np.loadtxt(input_dir / "davis/binary/X2.txt").astype("float32")
    y = np.loadtxt(input_dir / "davis/binary/y100.txt").astype("uint8")

    X1 = symmetrize(X1)
    X2 = symmetrize(X2)

    output_dir /= "davis"
    output_dir.mkdir()

    np.save(output_dir / "X1", X1)
    np.save(output_dir / "X2", X2)
    np.save(output_dir / "y", y)

    print("Transferred davis")


def transfer_kiba(input_dir: Path, output_dir: Path):
    X1 = pd.read_table(
        input_dir / "kiba/final/ligand_similarity.tsv", index_col=0
    ).values.astype("float32")

    X2 = pd.read_table(
        input_dir / "kiba/final/normalized_target_similarity.tsv", index_col=0
    ).values.astype("float32")

    y = pd.read_table(
        input_dir / "kiba/final/binary_affinity.tsv", index_col=0
    ).values.astype("uint8")

    X1 = symmetrize(X1)
    X2 = symmetrize(X2)

    output_dir /= "kiba"
    output_dir.mkdir()

    np.save(output_dir / "X1", X1)
    np.save(output_dir / "X2", X2)
    np.save(output_dir / "y", y)

    print("Transferred kiba")


def transfer_mirtarbase(input_dir: Path, output_dir: Path):
    X1 = pd.read_table(
        input_dir / "miRNA/final/normalized_mirna_similarity.tsv", index_col=0
    ).values.astype("float32")

    X2 = pd.read_table(
        input_dir / "miRNA/final/normalized_target_similarity.tsv", index_col=0
    ).values.astype("float32")

    y = pd.read_table(
        input_dir / "miRNA/final/interaction_matrix.tsv", index_col=0
    ).values.astype("uint8")

    X1 = symmetrize(X1)
    X2 = symmetrize(X2)

    output_dir /= "mirtarbase"
    output_dir.mkdir()

    np.save(output_dir / "X1", X1)
    np.save(output_dir / "X2", X2)
    np.save(output_dir / "y", y)

    print("Transferred mirtarbase")


def transfer_npinter(input_dir: Path, output_dir: Path):
    X1 = pd.read_table(
        input_dir / "lncRNA/normalized_lncrna_similarity.tsv", index_col=0
    ).values.astype("float32")

    X2 = pd.read_table(
        input_dir / "lncRNA/normalized_target_similarity.tsv", index_col=0
    ).values.astype("float32")

    y = pd.read_table(
        input_dir / "lncRNA/interaction_matrix.tsv", index_col=0
    ).values.astype("uint8")

    X1 = symmetrize(X1)
    X2 = symmetrize(X2)

    output_dir /= "npinter"
    output_dir.mkdir()

    np.save(output_dir / "X1", X1)
    np.save(output_dir / "X2", X2)
    np.save(output_dir / "y", y)

    print("Transferred npinter")


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output-dir", type=click.Path(exists=False, path_type=Path), required=True
)
def main(input_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    transfer_nuclear_receptors(input_dir, output_dir)
    transfer_gpcr(input_dir, output_dir)
    transfer_ion_channels(input_dir, output_dir)
    transfer_enzymes(input_dir, output_dir)
    transfer_ern(input_dir, output_dir)
    transfer_srn(input_dir, output_dir)
    transfer_davis(input_dir, output_dir)
    transfer_kiba(input_dir, output_dir)
    transfer_mirtarbase(input_dir, output_dir)
    transfer_npinter(input_dir, output_dir)


if __name__ == "__main__":
    main()
