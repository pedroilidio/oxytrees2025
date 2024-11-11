from bipartite_learn.base import BaseBipartiteEstimator

"""Distance Weighted Neighbors Regression."""

# Author: Pedro Ilídio <ilidio@alumni.usp.br>
# Adapted from scikit-learn.

# License: BSD 3 clause (C) Pedro Ilídio

import numpy as np

from joblib import effective_n_jobs
from sklearn.neighbors._base import (
    _get_weights,
    _check_precomputed,
    NeighborsBase,
    KNeighborsMixin,
)
from sklearn.base import RegressorMixin, _fit_context
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils._param_validation import StrOptions
from bipartite_learn.neighbors import WeightedNeighborsRegressor


class KroneckerWeightedNeighbors(
    KNeighborsMixin,
    RegressorMixin,
    NeighborsBase,
    BaseBipartiteEstimator,
):
    _parameter_constraints: dict = {
        **NeighborsBase._parameter_constraints,
        "weights": [StrOptions({"distance", "similarity"}), callable, None],
    }
    _parameter_constraints.pop("radius")
    _parameter_constraints.pop("n_neighbors")
    _parameter_constraints.pop("leaf_size")
    _parameter_constraints.pop("algorithm")

    def __init__(
        self,
        *,
        weights="distance",
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=None,
            algorithm="brute",
            leaf_size=None,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self.weights = weights

    def _more_tags(self):
        # For cross-validation routines to split data correctly
        return {"pairwise": self.metric == "precomputed"}

    def _validate_params(self):
        if (
            self.weights == "similarity"
            and self.metric != "precomputed"
            and isinstance(self.metric, str)
        ):
            raise ValueError(
                "weights='similarity' is not supported for string-valued"
                f" metrics other than 'precomputed' ({self.metric=!r})",
            )
        super()._validate_params()

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        # HACK
        rows_knn = WeightedNeighborsRegressor(
            weights=self.weights,
            p=self.p,
            metric=self.metric,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        ).fit(X[0], y)

        self._fit_X_rows = rows_knn._fit_X
        self.n_rows_fit_ = rows_knn.n_samples_fit_

        cols_knn = WeightedNeighborsRegressor(
            weights=self.weights,
            p=self.p,
            metric=self.metric,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        ).fit(X[1], y.T)

        self._fit_X_cols = cols_knn._fit_X
        self.n_cols_fit_ = cols_knn.n_samples_fit_

        self.n_neighbors = (X[0].shape[0], X[1].shape[0])
        self.n_samples_fit_ = self.n_rows_fit_ * self.n_cols_fit_
        self._fit_X = [self._fit_X_rows, self._fit_X_cols]
        self._y = y

        return self

    def predict(self, X):
        W = [self._check_weights(Xi, Xi_fit) for Xi, Xi_fit in zip(X, self._fit_X)]
        y_pred = W[0] @ self._y @ W[1].T
        return y_pred.reshape(-1)
    
    def _check_weights(self, Xi, Xi_fit):
        if self.metric == "precomputed":
            dist = _check_precomputed(Xi)
        else:
            Xi = self._validate_data(
                Xi,
                accept_sparse="csr",
                reset=False,
                order="C",
            )

            dist = pairwise_distances(
                Xi,
                Xi_fit,
                metric=self.effective_metric_,
                n_jobs=effective_n_jobs(self.n_jobs),
                **self.effective_metric_params_,
            )

        if self.weights == "similarity":
            weights = dist
        else:
            # Default weights is "distance", no more "uniform"
            weights = _get_weights(dist, self.weights or "distance")

        weights = weights.copy()

        # Normalize the weights to sum up to 1 on each row
        denom = np.sum(weights, axis=1, keepdims=True)
        mask = (denom == 0).reshape(-1)

        weights[mask] = 1
        denom[mask] = weights.shape[1]

        weights /= denom

        return weights