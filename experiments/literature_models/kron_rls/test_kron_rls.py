import pytest
import numpy as np
from scipy import linalg
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils._testing import assert_allclose
from bipartite_learn import datasets

import kron_rls


@pytest.fixture
def data():
    X, y = datasets.NuclearReceptorsLoader().load()
    X = [(x + x.T) / 2 for x in X]
    return X, y


@pytest.fixture
def large_data():
    X, y = datasets.EnzymesLoader().load()
    X = [(x + x.T) / 2 for x in X]
    return X, y


def test_vs_naive_rlskron(data):
    test_size = 20
    X, y = data
    X_test = [x[:test_size] for x in X]
    y_test = y[:test_size, :test_size]

    naive = kron_rls.NaiveKronRLSRegressor()
    naive.fit(X, y)
    naive_pred = naive.predict(X_test)
    assert naive_pred.shape == (y_test.size,)
    assert average_precision_score(y_test.ravel(), naive_pred) > 0.9

    estimator = kron_rls.KronRLSRegressor()
    estimator.fit(X, y)
    pred = estimator.predict(X_test)
    assert pred.shape == (y_test.size,)

    assert_allclose(naive_pred, pred)


def test_rlskron(large_data):
    X, y = large_data
    estimator = kron_rls.KronRLSRegressor()
    estimator.fit(X, y)
    pred = estimator.predict([X[0], X[1]])
    assert pred.shape == (len(X[0]) * len(X[1]),)
    assert (X[0] @ estimator.coef_ @ X[1].T).shape == y.shape

    ap = average_precision_score(y.ravel(), pred)
    auroc = roc_auc_score(y.ravel(), pred)
    print(f"Density: {y.mean()}, AP: {ap}, AUROC: {auroc}")
    assert ap > 0.99
    assert auroc > 0.99
