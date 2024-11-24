from pathlib import Path

import numpy as np
import joblib
from scipy.stats import loguniform
from sklearn.base import RegressorMixin, MetaEstimatorMixin, clone
from bipartite_learn.base import BaseBipartiteEstimator, BaseMultipartiteSampler
from bipartite_learn.matrix_factorization import NRLMFClassifier
from bipartite_learn.model_selection import (
    MultipartiteRandomizedSearchCV,
    make_multipartite_kfold,
)

class ProbaRegressor(MetaEstimatorMixin, BaseBipartiteEstimator, RegressorMixin):
    def __init__(self, estimator):
        self.estimator = estimator
    
    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self
    
    def predict(self, X):
        proba = self.estimator_.predict_proba(X)
        if isinstance(proba, list):  # Multilabel format
            return np.hstack([p[:, -1] for p in proba])
        return proba[:, -1]

class ProbaSampler(MetaEstimatorMixin, BaseMultipartiteSampler):
    def __init__(self, estimator):
        self.estimator = estimator
    
    def _fit_resample(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        proba = self.estimator_.predict_proba(X)

        if isinstance(proba, list):  # Multilabel format
            return X, np.hstack([p[:, -1] for p in proba])
        return X, proba[:, -1]


RSTATE = 0
N_JOBS = 1

kfold = make_multipartite_kfold(
    n_parts=2,  # Bipartite
    cv=2,
    shuffle=True,
    diagonal=False,
    random_state=RSTATE,
)

# The original proposal cannot be used:
# common_param_options = [2**-2, 2**-1, 1, 2]  # 18_432 parameter combinations!
common_param_options = loguniform(2**-2, 2)

nrlmf_grid = MultipartiteRandomizedSearchCV(
    NRLMFClassifier(random_state=RSTATE),
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
    scoring="roc_auc",
    cv=kfold,
    refit=True,
    n_jobs=N_JOBS,
    n_iter=100,
    verbose=1,
    random_state=RSTATE,
    pairwise=True,
)

nrlmf_regressor = ProbaRegressor(nrlmf_grid)
nrlmf_sampler = ProbaSampler(nrlmf_grid)
