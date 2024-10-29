"""Utilities for building similarity matrices of biological sequences or chemical
compounds.
"""
# Author: Pedro Ilidio <pedrilidio@gmail.com>, 2023
# License: BSD 3 clause
from typing import Any, Callable, Sequence
from itertools import combinations_with_replacement
import warnings
import pandas as pd
import numpy as np
from Bio.Align import substitution_matrices, PairwiseAligner
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.DataStructs
from joblib import Parallel, delayed


def normalize_similarity_matrix(similarity_matrix: np.ndarray):
    """Normalize Smith-Waterman scores.

    Normalize values of a similarity matrix composed of alignment scores to the
    0-1 interval.
    
    Parameters
    ----------
    similarity_matrix : np.ndarray
        Square similarity matrix between sequences.
    """
    diag = np.diag(similarity_matrix.values)
    denom = np.sqrt(diag[:, None] * diag[None, :])
    return similarity_matrix / denom


def compute_similarity_matrix(
    seqs: pd.Series | Sequence,
    similarity_func: Callable,
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
        Number of parallel jobs to run, by default 1.

    Returns
    -------
    pd.DataFrame
        A similarity matrix.
    """
    seqs = pd.Series(seqs)

    def compute_similarity(i, j):
        return i, j, similarity_func(seqs[i], seqs[j])

    total = (len(seqs) * (len(seqs) + 1)) // 2
    batch_size = int(np.ceil(total/n_jobs))

    print(f"Computing {total} pairwise similarities...")
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
    # Remove sequences with non-standard amino acids/nucleotides
    seqs = seqs[seqs.apply(lambda s: all(l in aligner.alphabet for l in s))]
    if seqs.empty:
        raise ValueError(
            "Sequences seem to use a different alphabet than the aligner."
        )
    return compute_similarity_matrix(seqs, aligner.score, **kwargs)


def compute_compound_similarity_matrix(smiles, **kwargs):
    """Calculate the ECPF4 compound similarity between SMILES strings.

    Parameters
    ----------
    smiles : pd.Series
        SMILES strings.
    n_jobs : int, optional
        Number of parallel jobs to run, by default 1.

    Returns
    -------
    pd.DataFrame
        Pairwise ECPF4 compound similarity between the SMILES strings.
    """
    molecules = smiles.apply(Chem.MolFromSmiles)
    fpgen = AllChem.GetMorganGenerator(radius=2)  # ECPF4 fingerprint

    if molecules.isna().any():
        warnings.warn(f"Could not parse the SMILES:\n{smiles[molecules.isna()]}")
        molecules = molecules.dropna()

    fingerprints = molecules.apply(fpgen.GetFingerprint)

    # Pickling-safe Tanimoto similarity function (needed for parallelization)
    class PickleableTanimotoSimilarity:
        def __call__(self, fp1, fp2):
            module = __import__(
                name="rdkit.DataStructs",
                globals={"rdkit.DataStructs": rdkit.DataStructs},
                fromlist=["TanimotoSimilarity"],
            )
            return module.TanimotoSimilarity(fp1, fp2)
    
    return compute_similarity_matrix(
        fingerprints,
        PickleableTanimotoSimilarity(),
        **kwargs,
    )
