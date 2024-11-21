from pathlib import Path
import pandas as pd

SEQS_PATH = Path("/home/pedro/mestrado/biomal_repo/scripts/data_fetching/seqs.csv")
OUTPUT_PATH = Path("./gene_utr3_sequences.tsv")

print(f"Loading sequences from {SEQS_PATH}.")
seqs = pd.read_csv(SEQS_PATH, usecols=("entrez_id", "three_prime_UTR_exons")).rename(
    columns={"three_prime_UTR_exons": "sequence", "entrez_id": "gene_id"}
)
seqs = seqs.dropna(subset=["sequence"])

print(f"Loaded {len(seqs)} sequences. Processing...")
seqs["sequence"] = seqs.sequence.str.upper()  # Ignore soft-masked bases.

print(f"Saving sequences to {OUTPUT_PATH}.")
seqs.to_csv(OUTPUT_PATH, sep="\t", index=False)
print("Done.")
