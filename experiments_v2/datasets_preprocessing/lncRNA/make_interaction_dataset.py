from argparse import ArgumentParser
from typing import Sequence
from pathlib import Path
from itertools import combinations_with_replacement
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Align import substitution_matrices, PairwiseAligner
from joblib import Parallel, delayed


def read_fasta_as_tsv(path):
    seqs = SeqIO.parse(Path(path), 'fasta')
    seqs = pd.Series(SeqIO.to_dict(seqs))
    return seqs.apply(lambda s: str(s.seq))


def normalize_similarity_matrix(similarity_matrix: pd.DataFrame):
    """Normalize Smith-Waterman scores.

    Normalize values of a similarity matrix composed of alignment scores to the
    0-1 interval.

    Parameters
    ----------
    similarity_matrix : pd.DataFrame
        Square similarity matrix between sequences.
    """
    diag = np.diag(similarity_matrix)
    denom = np.sqrt(diag[:, None] * diag[None, :])
    return similarity_matrix / denom


def compute_similarity_matrix(
    seqs: pd.Series | Sequence[str],
    aligner: PairwiseAligner,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compute a similarity matrix for the given sequences.

    Parameters
    ----------
    seqs : pd.Series | Sequence[str]
        A series or list-like of biological sequences.
    aligner : PairwiseAligner
        A Biopython pairwise aligner.
    n_jobs : int, optional
        Number of jobs to run in parallel, by default 1.
        (See joblib.Parallel documentation for more information.)

    Returns
    -------
    pd.DataFrame
        A similarity matrix.
    """
    seqs = pd.Series(seqs)
    if aligner.alphabet is not None:
        # Remove sequences with non-standard amino acids / nucleotides
        seqs = seqs[seqs.apply(lambda s: all(l in aligner.alphabet for l in s))]
        if seqs.empty:
            raise ValueError(
                "Sequencs seem to use a different alphabet than the aligner."
            )

    def compute_similarity(i, j):
        return i, j, aligner.score(seqs[i], seqs[j])

    total = (len(seqs) * (len(seqs) + 1)) // 2
    batch_size = int(np.ceil(total/n_jobs))

    print(f"Computing {total} pairwise alignments...")
    records = Parallel(n_jobs=n_jobs, batch_size=batch_size, verbose=5)(
        delayed(compute_similarity)(i, j)
        for i, j in combinations_with_replacement(seqs.index, 2)
    )

    print("Building similarity matrix...", end=" ")
    similarity_matrix = pd.DataFrame.from_records(records, columns=["i", "j", "score"])
    similarity_matrix = similarity_matrix.pivot(index="i", columns="j", values="score")
    # Fill in the lower triangle
    similarity_matrix = similarity_matrix.combine_first(similarity_matrix.T)
    # Fix possible ordering change due to parallelization
    similarity_matrix = similarity_matrix.loc[seqs.index, seqs.index]
    print("Done.")

    return similarity_matrix


def compute_protein_similarity_matrix(seqs, **kwargs):
    """Calculate the Smith-Waterman protein similarity between sequences.

    Parameters
    ----------
    seqs : pd.Series | Sequence[str]
        A series or list-like of protein sequences.
    n_jobs : int, optional
        Number of parallel jobs to run, by default 1.

    Returns
    -------
    pd.DataFrame
        Pairwise Smith-Waterman alignment scores between the sequences.
    """
    aligner = PairwiseAligner(
        mode="global",
        substitution_matrix=substitution_matrices.load("BLOSUM62"),
        open_gap_score=0,
        extend_gap_score=0,
    )
    return compute_similarity_matrix(seqs, aligner, **kwargs)


def compute_nucleotide_similarity_matrix(seqs, **kwargs):
    # Default parameters for BLASTN used by Biopython:
    # open_gap_score=-7,
    # extend_gap_score=-2,
    aligner = PairwiseAligner(
        mode="global",
        substitution_matrix=substitution_matrices.load("BLASTN"),
        open_gap_score=0,  # No penalties ensures positive scores (TODO: test)
        extend_gap_score=0,
    )
    return compute_similarity_matrix(seqs, aligner, **kwargs)


def drop_single_interactions(interactions, max_iter=15, min_lncrna=2, min_target=2):
    n_targets = len(interactions.tarID.unique())
    n_lncrna = len(interactions.ncID.unique())
    for _ in range(max_iter):
        lncrna_counts = interactions.ncID.value_counts()
        interactions = interactions[
            interactions.ncID.isin(lncrna_counts[lncrna_counts >= min_lncrna].index)
        ]
        target_counts = interactions.tarID.value_counts()
        interactions = interactions[
            interactions.tarID.isin(target_counts[target_counts >= min_target].index)
        ]
        n_remaining_targets = len(interactions.tarID.unique())
        n_remaining_lncrna = len(interactions.ncID.unique())
        print(f'Dropped {n_lncrna - n_remaining_lncrna} lncRNAs.')
        print(f'Dropped {n_targets - n_remaining_targets} targets.')

        if n_remaining_targets == n_targets and n_remaining_lncrna == n_lncrna:
            break
        n_targets = n_remaining_targets
        n_lncrna = n_remaining_lncrna

    print(f'Remaining {n_remaining_lncrna} lncRNAs and {n_remaining_targets} targets.')
    return interactions


def make_interaction_dataset(
    interactions_table_input: Path,
    lncrna_fasta_input: Path,
    target_fasta_input: Path,
    found_interactions_output: Path,
    lncrna_output: Path,
    target_output: Path,
    lncrna_similarity_output: Path,
    target_similarity_output: Path,
    norm_target_output: Path,
    norm_lncrna_similarity_output: Path,
    interaction_matrix_output: Path,
    n_jobs: int = 1,
):
    # Read interactions table (row example: NONHSAT000010  P0DTD1)
    interactions = pd.read_table(interactions_table_input)
    print(f"Loaded {len(interactions)} interactions.")

    # Drop duplicate interactions
    print(f"Dropping {interactions.duplicated().sum()} duplicate interactions.")
    interactions = interactions.drop_duplicates()

    # Iteratively filter single interactions
    # TODO: These numbers are arbitrary, we should set them as parameters.
    interactions = drop_single_interactions(interactions, min_lncrna=50, min_target=2)

    # Process lncRNA sequences
    lncrna_seqs = read_fasta_as_tsv(lncrna_fasta_input)
    # Remove transcript numbers from IDs (e.g. 'NONHSAT000010.2' -> 'NONHSAT000010')
    lncrna_seqs.index = lncrna_seqs.index.str.split('.', n=1).str[0]
    # Keep only the longest transcript for each lncRNA, since NPInter does not
    # specify which transcript is used.
    lncrna_seqs = lncrna_seqs.sort_values(key=lambda a: a.str.len(), ascending=True)
    lncrna_seqs = lncrna_seqs[~lncrna_seqs.index.duplicated(keep='last')]
    # Rename IDs to match those in the interactions table
    lncrna_seqs.index = lncrna_seqs.index.str.replace('NONHSAT', 'NONHSAG')
    # Select only the lnRNAs that are in the interactions table
    lncrna_seqs = lncrna_seqs[lncrna_seqs.index.isin(interactions.ncID.unique())]
    lncrna_seqs = lncrna_seqs.str.upper()  # Ignore soft-masking
    lncrna_seqs.index.name = 'ncID'
    lncrna_seqs.name = 'lncrna_seq'

    # Process target protein sequences
    target_seqs = read_fasta_as_tsv(target_fasta_input)
    # Get UniProt IDs from FASTA headers (e.g. 'sp|P0DTD1|SPIKE_SARS2' -> 'P0DTD1')
    target_seqs.index = target_seqs.index.str.split('|', n=2).str[1]
    target_seqs = target_seqs[target_seqs.index.isin(interactions.tarID.unique())]
    target_seqs.index.name = 'tarID'
    target_seqs.name = 'target_seq'

    # Filter interactions table to only include interactions with found sequences
    found_interactions = interactions[
        interactions.ncID.isin(lncrna_seqs.index)
        & interactions.tarID.isin(target_seqs.index)
    ]
    found_interactions.to_csv(found_interactions_output, sep='\t', index=False)
    print(f'Found interactions saved to {found_interactions_output}.')

    lncrna_seqs = lncrna_seqs.loc[found_interactions.ncID.unique()]
    lncrna_seqs.to_csv(lncrna_output, sep='\t')
    print(f'lncRNA sequences saved to {lncrna_output}.') 

    target_seqs = target_seqs.loc[found_interactions.tarID.unique()]
    target_seqs.to_csv(target_output, sep='\t')
    print(f'Target protein sequences saved to {target_output}.')

    # Make interaction matrix
    interaction_matrix = pd.crosstab(
        index=found_interactions.ncID,
        columns=found_interactions.tarID,
    )
    interaction_matrix = interaction_matrix.loc[lncrna_seqs.index, target_seqs.index]

    interaction_matrix.to_csv(interaction_matrix_output, sep='\t')
    print(f'Interaction matrix saved to {interaction_matrix_output}.')

    # Compute similarity matrices
    print("Calculating lncRNA-lncRNA similarities...")
    lncrna_similarity_matrix = compute_nucleotide_similarity_matrix(
        lncrna_seqs,
        n_jobs=n_jobs,
    )
    lncrna_similarity_matrix.to_csv(lncrna_similarity_output, sep='\t')
    print(f'lncRNA similarity matrix saved to {lncrna_similarity_output}.')

    print("Calculating protein-protein similarities...")
    target_similarity_matrix = compute_protein_similarity_matrix(
        target_seqs,
        n_jobs=n_jobs,
    )
    target_similarity_matrix.to_csv(target_similarity_output, sep='\t')
    print(f'Target protein similarity matrix saved to {target_similarity_output}.')

    norm_target_similarity_matrix = normalize_similarity_matrix(target_similarity_matrix)
    norm_target_similarity_matrix.to_csv(norm_target_output, sep='\t')
    print(f'Normalized target protein similarity matrix saved to {norm_target_output}.')
   
    norm_lncrna_similarity_matrix = normalize_similarity_matrix(lncrna_similarity_matrix)
    norm_lncrna_similarity_matrix.to_csv(norm_lncrna_similarity_output, sep='\t')
    print(f'Normalized lncRNA similarity matrix saved to {norm_lncrna_similarity_output}.')


def main():
    interactions_table_input = Path('lncrna_protein_interactions.tsv')
    lncrna_fasta_input = Path('NONCODEv6_human.fa')
    target_fasta_input = Path(
        'uniprot-compressed_true_download_true_format_fasta-2023.06.26-21.08.55.31.fasta'
    )
    found_interactions_output = Path('found_interactions.tsv')
    lncrna_output = Path('lncrna_seqs.tsv')
    target_output = Path('target_seqs.tsv')
    lncrna_similarity_output = Path('lncrna_similarity.tsv')
    target_similarity_output = Path('target_similarity.tsv')
    norm_target_output = Path('normalized_target_similarity.tsv')
    norm_lncrna_similarity_output = Path('normalized_lncrna_similarity.tsv')
    interaction_matrix_output = Path('interaction_matrix.tsv')

    parser = ArgumentParser(
        description=(
            'Format lncRNA-protein interaction data as a bipartite edge-prediction task.'
            '\nThis script will filter the interactions table to only include interactions'
            ' where both the lncRNA and target protein sequences are found in the'
            ' respective FASTA files. It will also compute similarity matrices for the'
            ' lncRNA and target protein sequences using scores of Smith-Waterman'
            ' alignments, and normalize the similarity matrices to have values between 0'
            ' and 1.'
            ' Finally, it will output a binary matrix of interactions between lncRNAs'
            ' and target proteins.'
        )
    )

    parser.add_argument(
        '--interactions_table_input',
        type=Path,
        default=interactions_table_input,
        help='Path to the interactions table.',
    )
    parser.add_argument(
        '--lncrna_fasta_input',
        type=Path,
        default=lncrna_fasta_input,
        help='Path to the lncRNA FASTA file.',
    )
    parser.add_argument(
        '--target_fasta_input',
        type=Path,
        default=target_fasta_input,
        help='Path to the target protein FASTA file.',
    )
    parser.add_argument(
        '--found_interactions_output',
        type=Path,
        default=found_interactions_output,
        help='Path to save the found interactions table.',
    )
    parser.add_argument(
        '--lncrna_output',
        type=Path,
        default=lncrna_output,
        help='Path to save the lncRNA sequences.',
    )
    parser.add_argument(
        '--target_output',
        type=Path,
        default=target_output,
        help='Path to save the target protein sequences.',
    )
    parser.add_argument(
        '--lncrna_similarity_output',
        type=Path,
        default=lncrna_similarity_output,
        help='Path to save the lncRNA similarity matrix.',
    )
    parser.add_argument(
        '--target_similarity_output',
        type=Path,
        default=target_similarity_output,
        help='Path to save the target protein similarity matrix.',
    )
    parser.add_argument(
        '--norm_target_output',
        type=Path,
        default=norm_target_output,
        help='Path to save the normalized target protein similarity matrix.',
    )
    parser.add_argument(
        '--norm_lncrna_similarity_output',
        type=Path,
        default=norm_lncrna_similarity_output,
        help='Path to save the normalized lncRNA similarity matrix.',
    )
    parser.add_argument(
        '--interaction_matrix_output',
        type=Path,
        default=interaction_matrix_output,
        help='Path to save the interaction matrix.',
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=1,
        help='Number of jobs to use for similarity calculation.',
    )
    args = parser.parse_args()

    make_interaction_dataset(
        interactions_table_input=args.interactions_table_input,
        lncrna_fasta_input=args.lncrna_fasta_input,
        target_fasta_input=args.target_fasta_input,
        found_interactions_output=args.found_interactions_output,
        lncrna_output=args.lncrna_output,
        target_output=args.target_output,
        lncrna_similarity_output=args.lncrna_similarity_output,
        target_similarity_output=args.target_similarity_output,
        norm_target_output=args.norm_target_output,
        norm_lncrna_similarity_output=args.norm_lncrna_similarity_output,
        interaction_matrix_output=args.interaction_matrix_output,
        n_jobs=args.n_jobs,
    )


if __name__ == '__main__':
    main()