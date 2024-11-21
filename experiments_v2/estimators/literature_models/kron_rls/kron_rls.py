import numpy as np
from scipy import linalg
from sklearn.base import RegressorMixin
from bipartite_learn.base import BaseBipartiteEstimator


class KronRLSRegressor(BaseBipartiteEstimator, RegressorMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y, **fit_params):
        # TODO: subtract Y mean of each column
        X1, X2 = X
        U1, s1, V1t = linalg.svd(X1, full_matrices=False)
        U2, s2, V2t = linalg.svd(X2, full_matrices=False)
        s = 1 / (np.outer(s1, s2) + self.alpha)
        self.coef_ = U1 @ (s * (V1t @ y @ V2t.T)) @ U2.T
        return self

    def predict(self, X):
        X1, X2 = X
        return (X1 @ self.coef_ @ X2.T).reshape(-1)


class NaiveKronRLSRegressor(BaseBipartiteEstimator, RegressorMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X1, X2 = X
        self.coef_ = linalg.inv((
            linalg.kron(X1, X2)
            + self.alpha * np.eye(X1.shape[0] * X2.shape[0])
        )) @ y.reshape(-1, 1)

    def predict(self, X):
        return (linalg.kron(*X) @ self.coef_).reshape(-1)
