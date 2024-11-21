import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import FunctionTransformer
from bipartite_learn.ensemble import BipartiteExtraTreesRegressor
from bipartite_learn.pipeline import make_multipartite_pipeline

from .model_forests.bipartite_model_trees import BipartiteModelForestRegressor
from .model_forests.dwnn import KroneckerWeightedNeighbors


def max_similarity(X):
    max_indices = np.argmax(X, axis=1)
    result = np.zeros_like(X)
    result[:, max_indices] = 1
    return result


def conditional_uniform(X):
    return (X == 1).astype(X.dtype)


uniform_bxt_bgso = BipartiteExtraTreesRegressor(
    bipartite_adapter="gmosa",
    criterion="squared_error_gso",
    n_estimators=1000,
    min_rows_leaf=5,
    min_cols_leaf=5,
    max_samples=None,
    bootstrap=False,
    random_state=0,
    verbose=10,
)

dwnn_max_bxt_bgso = BipartiteModelForestRegressor(
    estimator=clone(uniform_bxt_bgso),
    leaf_estimator=KroneckerWeightedNeighbors(
        metric="precomputed", weights=max_similarity,
    ),
    pairwise=True,
)

dwnn_conditional_uniform_bxt_bgso = BipartiteModelForestRegressor(
    estimator=clone(uniform_bxt_bgso),
    leaf_estimator=KroneckerWeightedNeighbors(
        metric="precomputed", weights=conditional_uniform,
    ),
    pairwise=True,
)

dwnn_similarities_bxt_bgso = BipartiteModelForestRegressor(
    estimator=clone(uniform_bxt_bgso),
    leaf_estimator=KroneckerWeightedNeighbors(
        metric="precomputed", weights="similarity"
    ),
    pairwise=True,
)

dwnn_square_bxt_bgso = make_multipartite_pipeline(
    # NOTE: One could also apply the function as the `weights` parameter in the
    # leaf_estimator, but since trees are invariant to monotonic transformations,
    # it is more efficient to apply the transformation here.
    FunctionTransformer(np.square),  
    clone(dwnn_similarities_bxt_bgso),
)

dwnn_softmax_bxt_bgso = make_multipartite_pipeline(
    # NOTE: See previous note.
    FunctionTransformer(np.exp),
    clone(dwnn_similarities_bxt_bgso),
)
