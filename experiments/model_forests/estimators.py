import numpy as np
import bipartite_learn.ensemble
from bipartite_learn.pipeline import make_multipartite_pipeline
from sklearn.utils import check_random_state
from sklearn.base import clone
from sklearn.preprocessing import FunctionTransformer

import bipartite_positive_dropper
from wrappers import regressor_to_binary_classifier

from semisupervised_forests.estimators import ss_bxt_gso__md_size
from model_forests.bipartite_model_trees import BipartiteModelForestRegressor
from model_forests.dwnn import KroneckerWeightedNeighbors

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

bxt_gso__md_size__sq = make_multipartite_pipeline(
    FunctionTransformer(square),
    BipartiteModelForestRegressor(
        estimator=ss_bxt_gso__md_size,
        leaf_estimator=KroneckerWeightedNeighbors(
            metric="precomputed", weights="similarity"
        ),
        pairwise=True,
    ),
)
