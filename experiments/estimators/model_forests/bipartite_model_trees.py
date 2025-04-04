from numbers import Real

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._param_validation import HasMethods, Interval
from sklearn.exceptions import NotFittedError
from bipartite_learn.ensemble._forest import BaseMultipartiteForest

from .model_trees import ModelForestRegressor


class BipartiteModelTree(MetaEstimatorMixin, BaseEstimator):
    _estimator_params = (
        "n_outputs_",
        "n_features_in_",
        "tree_",
        "_validate_X_predict",
    )
    _parameter_constraints = {
        "estimator": [HasMethods(["apply", "fit"])],
        "leaf_estimator": [HasMethods(["fit", "predict"])],
        "pairwise": ["boolean"],
        "min_impurity": [Interval(Real, 0.0, None, closed="left")],
    }

    def __init__(self, estimator, leaf_estimator, pairwise=False, min_impurity=0.0):
        self.estimator = estimator
        self.leaf_estimator = leaf_estimator
        self.pairwise = pairwise
        self.min_impurity = min_impurity

    def _iter_leaf_indices(self, X):
        check_is_fitted(self)
        leaves = self.estimator_.apply(X)
        df = pd.DataFrame(leaves.reshape(X[0].shape[0], -1)).stack()

        for leaf_id, leaf_group in df.groupby(df):
            leaf_group = leaf_group.unstack()
            yield leaf_id, leaf_group.index.values, leaf_group.columns.values
    
    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, **fit_params):
        # Fit tree if not already fitted
        try:
            check_is_fitted(self.estimator)
        except NotFittedError:
            self.estimator_ = clone(self.estimator).fit(X, y, **fit_params)
        else:
            self.estimator_ = self.estimator

        if self.pairwise:
            self._leaf_training_indices = {}

        self.leaf_estimators_ = {}
        for leaf_id, leaf_rows, leaf_cols in self._iter_leaf_indices(X):

            # Fit leaf estimator only if leaf is not homogeneous
            if self.tree_.impurity[leaf_id] <= self.min_impurity:
                # FIXME: only works for regression
                self.leaf_estimators_[leaf_id] = self.tree_.value[leaf_id][0]
                continue

            # Slice X matrix to only include rows and columns in leaf
            if self.pairwise:
                self._leaf_training_indices[leaf_id] = (leaf_rows, leaf_cols)
                X_leaf = [
                    X[0][leaf_rows, :][:, leaf_rows],
                    X[1][leaf_cols, :][:, leaf_cols],
                ]
            else:
                X_leaf = [X[0][leaf_rows, :], X[1][leaf_cols, :]]

            # Fit leaf estimator
            self.leaf_estimators_[leaf_id] = clone(self.leaf_estimator).fit(
                X_leaf,
                y[leaf_rows, :][:, leaf_cols],
            )

        return self

    def predict(self, X, check_input=True):
        check_is_fitted(self)
        X = self.estimator_._validate_X_predict(X, check_input=check_input)

        if self.n_outputs_ > 1:
            out_shape = (X[0].shape[0], X[1].shape[0], self.n_outputs_)
        else:
            out_shape = (X[0].shape[0], X[1].shape[0])

        y_hat = np.empty(out_shape, dtype=np.float64)

        for leaf_id, leaf_rows, leaf_cols in self._iter_leaf_indices(X):
            leaf_estimator = self.leaf_estimators_[leaf_id]

            if isinstance(leaf_estimator, np.ndarray):  # leaf is homogeneous
                y_hat[np.ix_(leaf_rows, leaf_cols)] = leaf_estimator
                continue

            if self.pairwise:
                leaf_train_rows, leaf_train_cols = self._leaf_training_indices[leaf_id]
                X_leaf = [
                    X[0][leaf_rows, :][:, leaf_train_rows],
                    X[1][leaf_cols, :][:, leaf_train_cols],
                ]
            else:
                X_leaf = [X[0][leaf_rows, :], X[1][leaf_cols, :]]

            leaf_y_hat = (
                self.leaf_estimators_[leaf_id].predict(X_leaf)
            ).reshape(len(leaf_rows), len(leaf_cols))

            y_hat[np.ix_(leaf_rows, leaf_cols)] = leaf_y_hat

        return y_hat.reshape(-1)

    def __sklearn_is_fitted__(self):
        return hasattr(self, "estimator_")

    def __getattribute__(self, name):
        if name != "_estimator_params" and name in self._estimator_params:
            return getattr(self.estimator_, name)
        return super().__getattribute__(name)


class BipartiteModelForestRegressor(ModelForestRegressor, BaseMultipartiteForest):
    _model_tree_class = BipartiteModelTree
