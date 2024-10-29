import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._param_validation import HasMethods
from sklearn.exceptions import NotFittedError
from sklearn.ensemble._forest import ForestRegressor


class ModelTree(MetaEstimatorMixin, BaseEstimator):
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
    }

    def __init__(self, estimator, leaf_estimator, pairwise=False):
        self.estimator = estimator
        self.leaf_estimator = leaf_estimator
        self.pairwise = pairwise

    def _iter_leaf_indices(self, X):
        check_is_fitted(self, "estimator_")
        leaves = self.estimator_.apply(X)
        group_iter = pd.Series(range(len(leaves))).groupby(leaves)

        for leaf_id, leaf_group in group_iter:
            yield leaf_id, leaf_group.values

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
            for leaf_id, leaf_indices in self._iter_leaf_indices(X):
                self._leaf_training_indices[leaf_id] = leaf_indices
                self.leaf_estimators_[leaf_id] = clone(self.leaf_estimator).fit(
                    X[leaf_indices, :][:, leaf_indices], y[leaf_indices]
                )
        else:
            self.leaf_estimators_ = {
                leaf_id: clone(self.leaf_estimator).fit(
                    X[leaf_indices, :], y[leaf_indices]
                )
                for leaf_id, leaf_indices in self._iter_leaf_indices(X)
            }

        return self

    def predict(self, X, check_input=True):
        check_is_fitted(self)
        X = self.estimator_._validate_X_predict(X, check_input=check_input)

        if self.n_outputs_ > 1:
            out_shape = (X.shape[0], self.n_outputs_)
        else:
            out_shape = (X.shape[0],)

        y_hat = np.empty(out_shape, dtype=np.float64)

        for leaf_id, leaf_indices in self._iter_leaf_indices(X):
            if self.pairwise:
                X_leaf = X[leaf_indices, :][:, self._leaf_training_indices[leaf_id]]
            else:
                X_leaf = X[leaf_indices, :]

            y_hat[leaf_indices] = self.leaf_estimators_[leaf_id].predict(X_leaf)

        return y_hat

    def __sklearn_is_fitted__(self):
        return hasattr(self, "estimator_")

    def __getattribute__(self, name):
        if name != "_estimator_params" and name in self._estimator_params:
            return getattr(self.estimator_, name)
        return super().__getattribute__(name)


class ModelForestRegressor(ForestRegressor, MetaEstimatorMixin):
    _estimator_params = (
        "_validate_data",
        "n_outputs_",
        "n_features_in_",
        "n_jobs",
        "n_estimators",
        "verbose",
        "criterion",
    )
    _model_tree_class = ModelTree

    def __init__(self, estimator, leaf_estimator, pairwise=False):
        self.estimator = estimator
        self.leaf_estimator = leaf_estimator
        self.pairwise = pairwise

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, **fit_params):
        self.estimator_ = clone(self.estimator).fit(X, y, **fit_params)
        self.estimators_ = []
        for tree in self.estimator_.estimators_:
            model_tree = self._model_tree_class(
                tree,
                self.leaf_estimator,
                self.pairwise,
            )
            self.estimators_.append(model_tree.fit(X, y))
        return self

    def __sklearn_is_fitted__(self):
        return hasattr(self, "estimators_")

    def __getattribute__(self, name):
        if name != "_estimator_params" and name in self._estimator_params:
            return getattr(self.estimator_, name)
        return super().__getattribute__(name)
