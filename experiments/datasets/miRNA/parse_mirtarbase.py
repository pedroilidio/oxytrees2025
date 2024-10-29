import gzip
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from Bio import SeqIO


def parse_mirtarbase(
    mirtarbase_input: Path,
    mirbase_input: Path,
    interactions_output: Path,
    mirna_sequences_output: Path,
    target_ids_output: Path,
):
    # Load miRNA-target interactions from miRTarBase
    print("Loading miRTarBase data...")
    mirtarbase = pd.read_excel(
        mirtarbase_input,
        usecols=['miRNA', 'Target Gene (Entrez ID)'],
        dtype=str,
    )
    mirtarbase = mirtarbase.rename(
        columns={
            'miRNA': 'mirna_id',
            'Target Gene (Entrez ID)': 'target_gene_id',
        }
    )
    print(f"Loaded {len(mirtarbase)} interactions from {mirtarbase_input}.")

    # Load miRNA sequences from miRBase
    print("Loading miRBase data...")
    with gzip.open(mirbase_input, "rt") as handle:
        mirbase = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))
    mirbase = pd.Series(mirbase).rename_axis("mirna_id").rename("sequence")
    mirbase = mirbase.apply(lambda s: str(s.seq).upper())  # Ignore soft-masking
    print(f"Loaded {len(mirbase)} miRNAs from {mirbase_input}.")

    # Filter miRTarBase interactions to only those with miRNA sequences in miRBase
    print("Filtering miRTarBase interactions...")
    mirtarbase = mirtarbase[mirtarbase.mirna_id.isin(mirbase.index)]
    mirbase = mirbase[mirbase.index.isin(mirtarbase.mirna_id.unique())]
    target_genes_ids = pd.Series(mirtarbase.target_gene_id.unique())

    print("Saving data...")    
    # Save miRNA-target interactions
    mirtarbase.to_csv(
        interactions_output,
        sep='\t',
        index=False,
    )
    print(f"Saved {len(mirtarbase)} interactions to {interactions_output}.")
    # Save miRNA sequences
    mirbase.to_csv(
        mirna_sequences_output,
        sep='\t',
    )
    print(f"Saved {len(mirbase)} miRNAs to {mirna_sequences_output}.")
    # Save target gene Entrez ids
    target_genes_ids.to_csv(
        target_ids_output,
        sep='\t',
        header=False,
        index=False,
    )
    print(f"Saved {len(target_genes_ids)} target genes to {target_ids_output}.")
    print("Done.")


def main():
    parser = ArgumentParser(
        description=(
            "Parse miRTarBase data file into simpler files with interactions, miRNAs,"
            " and target genes."
        ),
    )
    parser.add_argument(
        "--mirtarbase-input",
        type=Path,
        help="Path to miRTarBase data file.",
        default="hsa_MTI.xlsx",
    )
    parser.add_argument(
        "--mirbase-input",
        type=Path,
        help="Path to miRBase data file.",
        default="mature.fa.gz",
    )
    parser.add_argument(
        "--interactions-output",
        type=Path,
        help="Path to output file with miRNA-target interactions.",
        default="interactions.tsv",
    )
    parser.add_argument(
        "--mirna-sequences-output",
        type=Path,
        help="Path to output file with miRNA sequences.",
        default="mirna_sequences.tsv",
    )
    parser.add_argument(
        "--target-ids-output",
        type=Path,
        help="Path to output file with target gene Entrez IDs (NCBI Gene database).",
        default="gene_ids.txt",
    )
    args = parser.parse_args()

    parse_mirtarbase(
        mirtarbase_input=args.mirtarbase_input,
        mirbase_input=args.mirbase_input,
        interactions_output=args.interactions_output,
        mirna_sequences_output=args.mirna_sequences_output,
        target_ids_output=args.target_ids_output,
    )


if __name__ == "__main__":
    main()
