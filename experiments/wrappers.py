import numpy as np

from sklearn.base import MetaEstimatorMixin, ClassifierMixin
from imblearn.pipeline import Pipeline

from bipartite_learn.wrappers import MultipartiteSamplerWrapper
from bipartite_learn.preprocessing import SymmetryEnforcer
from bipartite_learn.base import (
    BaseMultipartiteEstimator,
    BaseMultipartiteSampler,
)

class RegressorToBinaryClassifier(
    BaseMultipartiteEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        self.threshold_ = y.mean()
        self.classes_ = np.array([0, 1])
        self.estimator.fit(X=X, y=y, **fit_params)
        return self
    
    def predict_proba(self, X):
        pred = self.estimator.predict(X).reshape(-1, 1)
        return np.hstack((1 - pred, pred))

    def predict(self, X):
        return (self.estimator.predict(X) > self.threshold_).astype(int)


class RegressorAsSampler(
    BaseMultipartiteSampler,
    MetaEstimatorMixin,
):
    # Only multipartite, since fit_resample will check input.

    def __init__(self, estimator, keep_positives=True):
        self.estimator = estimator
        self.keep_positives = keep_positives

    def _fit_resample(self, X, y):
        if hasattr(self.estimator, "fit_predict"):
            yt = self.estimator.fit_predict(X, y)
        else:
            yt = self.estimator.fit(X, y).predict(X)

        yt = yt.reshape(len(X[0]), len(X[1]))

        if self.keep_positives:
            yt[y == 1] = 1

        return X, yt


class ClassifierAsSampler(
    BaseMultipartiteSampler,
    MetaEstimatorMixin,
):
    # Only multipartite, since fit_resample will check input.

    def __init__(self, estimator, keep_positives=True):
        self.estimator = estimator
        self.keep_positives = keep_positives

    def _fit_resample(self, X, y):
        if hasattr(self.estimator, "fit_predict_proba"):
            yt = self.estimator.fit_predict_proba(X, y)[:, 1]
        else:
            yt = self.estimator.fit(X, y).predict_proba(X)[:, 1]

        yt = yt.reshape(len(X[0]), len(X[1]))

        if self.keep_positives:
            yt[y == 1] = 1

        return X, yt


def regressor_to_binary_classifier(estimator):
    return RegressorToBinaryClassifier(estimator)