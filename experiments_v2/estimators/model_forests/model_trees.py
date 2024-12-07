from numbers import Real

import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._param_validation import HasMethods, Interval
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
        "min_impurity": [Interval(Real, 0.0, None, closed="left")],
    }

    def __init__(self, estimator, leaf_estimator, pairwise=False, min_impurity=0.0):
        self.estimator = estimator
        self.leaf_estimator = leaf_estimator
        self.pairwise = pairwise
        self.min_impurity = min_impurity

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

            # Fit leaf estimator only if leaf is not homogeneous
            if self.tree_.impurity[leaf_id] <= self.min_impurity:
                # FIXME: only works for regression
                self.leaf_estimators_[leaf_id] = self.tree_.value[leaf_id][0]
                continue
            if self.pairwise:
                self._leaf_training_indices[leaf_id] = leaf_indices
                X_leaf = X[leaf_indices, :][:, leaf_indices]
            else:
                X_leaf = X[leaf_indices, :]

            # Fit leaf estimator
            self.leaf_estimators_[leaf_id] = clone(self.leaf_estimator).fit(
                X_leaf, y[leaf_indices]
            )

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
            leaf_estimator = self.leaf_estimators_[leaf_id]

            if isinstance(leaf_estimator, np.ndarray):  # leaf is homogeneous
                y_hat[leaf_indices] = leaf_estimator
                continue

            if self.pairwise:
                X_leaf = X[leaf_indices, :][:, self._leaf_training_indices[leaf_id]]
            else:
                X_leaf = X[leaf_indices, :]

            y_hat[leaf_indices] = leaf_estimator.predict(X_leaf)

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

    def __init__(self, estimator, leaf_estimator, pairwise=False, min_impurity=0.0):
        self.estimator = estimator
        self.leaf_estimator = leaf_estimator
        self.pairwise = pairwise
        self.min_impurity = min_impurity

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, **fit_params):
        self.estimator_ = clone(self.estimator).fit(X, y, **fit_params)
        self.estimators_ = []
        for tree in self.estimator_.estimators_:
            model_tree = self._model_tree_class(
                tree,
                self.leaf_estimator,
                self.pairwise,
                self.min_impurity,
            )
            self.estimators_.append(model_tree.fit(X, y))
        return self
    
    def predict(self, X):
        def _predict_tree(tree):
            return tree.predict(X)

        predictions = joblib.Parallel(
            n_jobs=self.n_jobs,
            # return_as="generator_unordered",  # Not supported by multiprocessing
            backend="multiprocessing",  # Threads will not work with ModelTree
            verbose=self.verbose,
        )(
            joblib.delayed(_predict_tree)(tree)
            for tree in self.estimators_
        )

        summed_predictions = sum(predictions)
        return summed_predictions / self.n_estimators

    def __sklearn_is_fitted__(self):
        return hasattr(self, "estimators_")

    def __getattribute__(self, name):
        if name != "_estimator_params" and name in self._estimator_params:
            return getattr(self.estimator_, name)
        return super().__getattribute__(name)
