import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import pandas as pd
import numpy as np
from http.client import HTTPException
from Bio import Entrez, SeqIO
from tqdm import tqdm
from time import sleep, time
import io

GENE_TABLE_DTYPES = {
    "GeneID": str,
    "genomic_nucleotide_accession.version": str,
    "start_position_on_the_genomic_accession": int,
    "end_position_on_the_genomic_accession": int,
}


def rate_limited_iterator(iterable, max_per_second: int):
    """Wraps an iterator to yield its elements at a maximum rate.

    Parameters
    ----------
    iterable : Iterable
        The iterable to iterate over.
    max_per_second : int
        The maximum number of elements to yield per second.
    """
    t = time()
    count = 0
    for element in iterable:
        yield element
        count += 1
        if count >= max_per_second:
            elapsed = time() - t
            if elapsed < 1.0:
                sleep(1.0 - elapsed)
            t = time()
            count = 0


def load_gene_entrez_ids(gene_entrez_ids_input: Path) -> pd.Series:
    """Loads gene Entrez IDs from a simple text file with one ID per row."""
    data =  pd.read_csv(
        gene_entrez_ids_input,
        sep="\t",
        header=None,
        dtype=str,
    )
    return data.loc[:, 0]


def fasta_to_series(input_fasta):
    """Loads a fasta file into a pd.Series, dropping duplicated values.

    Parameters
    ----------
    input_fasta : Path
        Path to the fasta file.
    
    Returns
    -------
    pd.Series
        Series with gene IDs as index and sequences as values.
    """
    records = SeqIO.parse(input_fasta, "fasta")
    result = pd.Series(dtype=str).rename_axis("gene_id").rename("sequence")

    for record in tqdm(list(records)):
        # Ignore duplicates
        if record.id in result.index:
            warnings.warn(
                f"Found duplicate gene ID {record.id} in {input_fasta}. Ignoring."
            )
            continue
        # Upper to ignore soft-masking
        result[record.id] = str(record.seq).upper()

    return result


def fetch_gene_table(
    gene_entrez_ids: pd.Series,
    gene_table_output: Path,
    chunksize: int = 100,
) -> pd.DataFrame:
    """Fetches gene information from NCBI's gene database in tabular format.

    If the file already exists, it loads it from disk and fetches only the
    missing genes.

    Parameters
    ----------
    gene_entrez_ids : pd.Series
        Entrez IDs of the genes to fetch.
    gene_table_output : Path
        Path to save the gene table to.
    chunksize : int, optional
        Number of genes to fetch at a time, by default 100

    Returns
    -------
    pd.DataFrame
        The table containing gene information from NCBI Gene.
    """
    if gene_table_output.exists():
        print(f"Loading tabular gene data from {gene_table_output}...")
        gene_table = pd.read_table(
            gene_table_output,
            dtype=GENE_TABLE_DTYPES,
        )
        missing_genes = gene_entrez_ids[~gene_entrez_ids.isin(gene_table.GeneID)]
        print(f"Loaded {len(gene_table)} genes.")
    else:
        gene_table = pd.DataFrame()
        missing_genes = gene_entrez_ids

    if missing_genes.empty:
        return gene_table

    print(
        "Querying NCBI's gene database for information on "
        f" {len(missing_genes)} genes..."
    )
    n_chunks = max(len(missing_genes) // chunksize, 1)
    chunk_iterator = rate_limited_iterator(
        tqdm(np.array_split(missing_genes, n_chunks)),
        max_per_second=3,  # Comply to NCBI's limit of 3 requests per second.
    )
    try:  # Using try-finally to ensure gene_table is saved even if an error occurs
        for chunk in chunk_iterator:
            new_rows = fetch_gene(gene_ids=chunk)
            if new_rows is None:
                continue
            # Filter out rows that are not the current version of the gene.
            # TODO: fetch the current version instead of filtering
            if (new_rows.CurrentID != 0).any():
                warnings.warn(
                    "The following genes are not the current version and"
                    " will be ignored. Please provide their current IDs to"
                    f" include them.\n{new_rows[new_rows.CurrentID != 0]}"
                )
                new_rows = new_rows[new_rows.CurrentID == 0]
            old_len = len(new_rows)
            new_rows = new_rows.dropna(
                subset=[
                    "start_position_on_the_genomic_accession",
                    "end_position_on_the_genomic_accession",
                ]
            )
            if len(new_rows) != old_len:
                warnings.warn(
                    f"Found {old_len - len(new_rows)} genes with missing"
                    " genomic coordinates. These genes will be ignored."
                )
            new_rows = new_rows.astype(GENE_TABLE_DTYPES)
            gene_table = pd.concat([gene_table, new_rows])
    finally:
        gene_table.to_csv(gene_table_output, sep="\t", index=False)
        print(
            f"Saved tabular data for {len(gene_table)} genes to"
            f" {gene_table_output}."
        )

    return gene_table


def fetch_gene(gene_ids):
    try:
        with Entrez.efetch(
            db="gene",
            id=gene_ids,
            rettype="tabular",
            retmode="text",
        ) as handle:
            new_rows = pd.read_table(io.StringIO(handle.read()))
        return new_rows
    except HTTPException as e:
        warnings.warn(
            f"Info on genes {gene_ids.to_list()}"
            f" failed to be retrieved due to the following error:\n{e}"
        )
    return None


def fetch_gene_sequences(
    gene_table: pd.DataFrame, gene_sequences_output: Path, sequences_table_output: Path
) -> pd.Series:
    """Fetch gene sequences from NCBI's nuccore database.

    Gene information previously fetched from NCBI's gene database includes
    the chromosome and coordinates of the gene. This function uses that
    information to fetch the gene sequence from NCBI's nuccore database.

    If the sequences_table_output file already exists, it loads it from disk
    and fetches only the sequences not yet downloaded.

    Two outputs are generated, see Parameters for details.

    Parameters
    ----------
    gene_table : pd.DataFrame
        Table of gene information fetched from NCBI's gene database.
    gene_sequences_output : Path
        Path to save the gene sequences to.
    sequences_table_output : Path
        Path to save a table relating GeneIDs (not Nuccore) to their sequence.

    Returns
    -------
    pd.DataFrame
        Contents of the sequences_table_output file.
    """
    # Look for sequences already downloaded
    if sequences_table_output.exists():
        seq_table = pd.read_table(
            sequences_table_output,
            dtype=str,
        ).set_index("gene_id").sequence  # Set index after loading to ensure dtype=str

        missing_genes = gene_table[~gene_table.GeneID.isin(seq_table.index)]
        print(f"Loaded {len(seq_table)} gene sequences.")
    else:
        seq_table = pd.Series().rename_axis("gene_id").rename("sequence")
        missing_genes = gene_table
    
    if missing_genes.empty:
        return seq_table

    # NOTE: eFetch does not support selecting sub-sequences if multiple sequences
    # are passed, so we fetch them one at a time. The other option would be to fetch
    # each entire chromosome and slice them ourselves, but we choose not to,
    # as chromosomes are large (although few) and thus consume memory and bandwidth.
    # Fethcing one sequence at a time also allows easily adding the GeneID to
    # the FASTA record.
    print(
        f"Fetching {len(missing_genes)} gene sequences from NCBI's nuccore database"
        f" and saving to {gene_sequences_output}..."
    )
    gene_sequences_output.touch()
    gene_iterator = rate_limited_iterator(
        tqdm(missing_genes.iterrows(), total=len(missing_genes)),
        max_per_second=3,  # Comply to NCBI's limit of 3 requests per second.
    )

    with (
        ThreadPoolExecutor() as executor,
        gene_sequences_output.open("a") as fasta_file,
    ):
        futures = []
        # Using try-finally to ensure seq_table is saved even if an error occurs
        # (e.g. KeyboardInterrupt).
        try:
            print("Submitting requests...")
            for _, gene_data in gene_iterator:
                futures.append(executor.submit(fetch_nuccore, gene_data))
        finally:
            print("Collecting results...")
            for future in tqdm(as_completed(futures), total=len(missing_genes)):
                try:
                    fasta_record = future.result()
                except Exception as e:
                    print(f"Error fetching gene sequence: {e}")
                    fasta_record = None

                if fasta_record is not None:
                    fasta_file.write(fasta_record + "\n")

            print("Building gene sequences table...")
            seq_table = fasta_to_series(gene_sequences_output)
            print("Saving...")
            seq_table.to_csv(sequences_table_output, sep="\t")
            print(
                f"\nSaved {len(seq_table)} gene sequences to {sequences_table_output}."
            )

    return seq_table


def fetch_nuccore(gene_data: pd.Series) -> str | None:
    """Fetch a gene sequence from NCBI's nuccore database.

    Parameters
    ----------
    gene_data : pd.Series
        Row of gene_table containing the gene information.
    
    Returns
    -------
    str | None
        FASTA record of the gene sequence, or None if the sequence could not be
        fetched.
    """
    try:
        with Entrez.efetch(
            db="nuccore",
            id=gene_data["genomic_nucleotide_accession.version"],
            seq_start=gene_data["start_position_on_the_genomic_accession"],
            seq_stop=gene_data["end_position_on_the_genomic_accession"],
            # See https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.strand
            strand={"plus": "1", "minus": "2"}[gene_data.orientation],
            rettype="fasta",
            retmode="text",
        ) as handle:
            fasta = handle.read()

        # Test sequence length
        tab_len = (
            gene_data.end_position_on_the_genomic_accession
            - gene_data.start_position_on_the_genomic_accession
            + 1
        )
        seq_len = len(fasta.split('\n', maxsplit=1)[1].replace('\n', ''))
        if tab_len != seq_len:
            warnings.warn(
                f"Sequence length mismatch for gene {gene_data.GeneID}:"
                f" {tab_len} (from gene table) != {seq_len} (from sequence)"
            )

        return f">{gene_data.GeneID} {fasta.removeprefix('>')}"

    except HTTPException as e:
        warnings.warn(
            f"Gene {gene_data.GeneID} failed"
            f" to be retrieved due to the following error:\n{e}"
        )
    
    return None


def fetch_gene_sequences_pipeline(
    gene_entrez_ids_input: Path,
    gene_sequences_output: Path,
    gene_table_output: Path,
    sequences_table_output: Path,
    chunksize: int = 100,
    email: str = "",
):
    Entrez.email = email
    gene_entrez_ids = load_gene_entrez_ids(gene_entrez_ids_input)
    gene_table = fetch_gene_table(gene_entrez_ids, gene_table_output, chunksize)
    seq_table = fetch_gene_sequences(
        gene_table, gene_sequences_output, sequences_table_output
    )


def main():
    parser = ArgumentParser(
        description="Fetch gene sequences from NCBI.",
        formatter_class=ArgumentDefaultsHelpFormatter,  # Show defaults in help message
    )
    parser.add_argument(
        "--gene-ids-input",
        type=Path,
        help=(
            "Path to file containing target genes accession numbers for the NCBI"
            " Gene database."
        ),
        default=Path("gene_ids.txt"),
    )
    parser.add_argument(
        "--gene-sequences-output",
        type=Path,
        default=Path("gene_sequences.fasta"),
    )
    parser.add_argument(
        "--gene-table-output",
        type=Path,
        default=Path("gene_table.tsv"),
    )
    parser.add_argument(
        "--sequences-table-output",
        type=Path,
        default=Path("gene_sequences.tsv"),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        help="Number of genes to query at once.",
        default=100,
    )
    parser.add_argument(
        "--email",
        type=str,
        help="Email address to use for NCBI API.",
        required=True,
    )

    args = parser.parse_args()

    fetch_gene_sequences_pipeline(
        gene_entrez_ids_input=args.gene_ids_input,
        gene_sequences_output=args.gene_sequences_output,
        gene_table_output=args.gene_table_output,
        sequences_table_output=args.sequences_table_output,
        email=args.email,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
