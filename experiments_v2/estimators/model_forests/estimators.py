import numpy as np
from sklearn.base import clone, RegressorMixin, BaseEstimator, MetaEstimatorMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from bipartite_learn.ensemble import BipartiteExtraTreesRegressor
from bipartite_learn.pipeline import make_multipartite_pipeline
from bipartite_learn.wrappers import GlobalSingleOutputWrapper

from .bipartite_model_trees import BipartiteModelForestRegressor
from .dwnn import KroneckerWeightedNeighbors
from ..literature_models.kron_rls import KronRLSRegressor

def max_similarity(X):
    max_indices = np.argmax(X, axis=1)
    result = np.zeros_like(X)
    result[:, max_indices] = 1
    return result


def conditional_uniform(X):
    return (X == 1).astype(X.dtype)


class ProbaRegressor(BaseEstimator, RegressorMixin, MetaEstimatorMixin):
    def __init__(self, estimator):
        self.estimator = estimator
    
    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self
    
    def predict(self, X):
        return self.estimator_.predict_proba(X)[:, 1].reshape(-1)


dwnn = KroneckerWeightedNeighbors(
    metric="precomputed",
    weights=max_similarity,
)

dwnn_square = make_multipartite_pipeline(
    FunctionTransformer(np.square),  
    clone(dwnn),
)

uniform_bxt_bgso = BipartiteExtraTreesRegressor(
    bipartite_adapter="gmosa",
    criterion="squared_error_gso",
    n_estimators=200,
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

bxt_bgso_kronrls = BipartiteModelForestRegressor(
    estimator=clone(uniform_bxt_bgso),
    leaf_estimator=KronRLSRegressor(),
    pairwise=True,
)

bxt_bgso_logistic = BipartiteModelForestRegressor(
    estimator=clone(uniform_bxt_bgso),
    leaf_estimator=GlobalSingleOutputWrapper(ProbaRegressor(LogisticRegression())),
    pairwise=True,
)