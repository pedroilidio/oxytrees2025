import numpy as np
from bipartite_learn.pipeline import make_multipartite_pipeline
from sklearn.base import clone
from sklearn.preprocessing import FunctionTransformer
from bipartite_learn.preprocessing.monopartite import SymmetryEnforcer

import wrappers
from semisupervised_forests.estimators import ss_bxt_gso__md_size
from model_forests.bipartite_model_trees import BipartiteModelForestRegressor
from model_forests.dwnn import KroneckerWeightedNeighbors
from y_reconstruction.estimators import nrlmf_grid

COMMON_PARAMS = dict(
    bipartite_adapter="gmosa",
    criterion="squared_error_gso",
    n_estimators=1000,
    min_rows_leaf=5,
    min_cols_leaf=5,
    max_samples=None,
    bootstrap=False,
    random_state=0,
    verbose=10,
    n_jobs=4,
)


def square(X):
    return np.square(X)


ss_bxt_gso__md_size = clone(ss_bxt_gso__md_size).set_params(**COMMON_PARAMS)

nrlmf__bxt_gso__md_size__sq = make_multipartite_pipeline(
    SymmetryEnforcer(),
    wrappers.ClassifierAsSampler(nrlmf_grid, keep_positives=True),
    FunctionTransformer(square),
    BipartiteModelForestRegressor(
        estimator=ss_bxt_gso__md_size,
        leaf_estimator=KroneckerWeightedNeighbors(
            metric="precomputed", weights="similarity"
        ),
        pairwise=True,
    ),
)
