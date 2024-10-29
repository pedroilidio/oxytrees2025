from copy import deepcopy
import numpy as np

from scipy.stats import loguniform
from bipartite_learn.pipeline import make_multipartite_pipeline
from bipartite_learn.preprocessing.monopartite import (
    TargetKernelDiffuser,
    SymmetryEnforcer,
)
from bipartite_learn.matrix_factorization import (
    NRLMFClassifier,
    DNILMFClassifier,
)
from bipartite_learn.model_selection import (
    MultipartiteRandomizedSearchCV,
    make_multipartite_kfold,
)

import wrappers
import bipartite_positive_dropper


RSTATE = np.random.RandomState(0)

kfold_5_shuffle_diag = make_multipartite_kfold(
    n_parts=2,  # Bipartite
    cv=5,
    shuffle=True,
    diagonal=True,
    random_state=0,
)

# The original proposal cannot be used:
# common_param_options = [2**-2, 2**-1, 1, 2]  # 18_432 parameter combinations!
common_param_options = loguniform(2**-2, 2)

nrlmf_grid = MultipartiteRandomizedSearchCV(
    NRLMFClassifier(random_state=deepcopy(RSTATE)),
    param_distributions=dict(
        lambda_rows=common_param_options,
        lambda_cols=common_param_options,
        alpha_rows=common_param_options,
        alpha_cols=common_param_options,
        learning_rate=common_param_options,
        n_neighbors=[3, 5, 10],
        n_components_rows=[50, 100],
        # n_components_cols="same",
    ),
    scoring="average_precision",
    cv=deepcopy(kfold_5_shuffle_diag),
    refit=True,
    verbose=1,
    n_jobs=3,
    n_iter=100,
    random_state=0,
    pairwise=True,
)

nrlmf = nrlmf_grid


dnilmf_grid = MultipartiteRandomizedSearchCV(
    DNILMFClassifier(random_state=deepcopy(RSTATE)),
    param_distributions=dict(
        lambda_rows=common_param_options,
        lambda_cols=common_param_options,
        beta=[0.1, 0.2, 0.4, 0.5],
        gamma=[0.1, 0.2, 0.4, 0.5],
        learning_rate=common_param_options,
        n_neighbors=[3, 5, 10],
        n_components_rows=[50, 100],
    ),
    scoring="average_precision",
    cv=deepcopy(kfold_5_shuffle_diag),
    refit=True,
    verbose=1,
    n_jobs=3,
    n_iter=100,
    random_state=0,
    pairwise=True,
)

dnilmf = make_multipartite_pipeline(
    SymmetryEnforcer(),
    TargetKernelDiffuser(),
    dnilmf_grid,
    memory="/tmp",
)


def nrlmf_y_reconstruction_wrapper(
    estimator, drop=None, random_state=None,  **params,
):
    if drop is None:
        pipe = []
    else:
        pipe = [
            bipartite_positive_dropper.BipartitePositiveDropper(
                drop,
                random_state=random_state
            )
        ]
    pipe += [
        SymmetryEnforcer(),
        wrappers.ClassifierAsSampler(nrlmf_grid, keep_positives=True),
        estimator,
    ]
    return make_multipartite_pipeline(*pipe, memory="/tmp").set_params(**params)


def dnilmf_y_reconstruction_wrapper(
    estimator, drop=None, random_state=None,  **params,
):
    if drop is None:
        pipe = []
    else:
        pipe = [
            bipartite_positive_dropper.BipartitePositiveDropper(
                drop,
                random_state=random_state
            )
        ]
    pipe += [
        SymmetryEnforcer(),
        TargetKernelDiffuser(),
        wrappers.ClassifierAsSampler(dnilmf_grid, keep_positives=True),
        estimator,
    ]
    return make_multipartite_pipeline(*pipe, memory="/tmp").set_params(**params)
