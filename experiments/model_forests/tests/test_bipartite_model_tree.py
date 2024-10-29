from pathlib import Path
import sys

import pytest
from sklearn.metrics import mean_squared_error
from bipartite_learn.tree import (
    BipartiteDecisionTreeRegressor,
    BipartiteExtraTreeRegressor,
)
from bipartite_learn.neighbors import WeightedNeighborsRegressor
from bipartite_learn.wrappers import GlobalSingleOutputWrapper
from bipartite_learn.ensemble import BipartiteRandomForestRegressor
from bipartite_learn.datasets import NuclearReceptorsLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from bipartite_model_trees import BipartiteModelTree, BipartiteModelForestRegressor


@pytest.fixture(params=range(10))
def random_state(request):
    return request.param


@pytest.fixture()
def data():
    return NuclearReceptorsLoader().load()


@pytest.fixture()
def fitted_tree(random_state, data):
    XX, Y = data
    leaf_estimator = GlobalSingleOutputWrapper(
        WeightedNeighborsRegressor(),
    )
    tree = BipartiteDecisionTreeRegressor(
        min_rows_leaf=5,
        min_cols_leaf=7,
        bipartite_adapter="gmosa",
        criterion="squared_error_gso",
        random_state=random_state,
    )
    model_tree = BipartiteModelTree(tree, leaf_estimator)
    model_tree.fit(XX, Y)
    return model_tree


@pytest.fixture()
def fitted_forest(random_state, data):
    XX, Y = data
    leaf_estimator = GlobalSingleOutputWrapper(
        WeightedNeighborsRegressor(),
    )
    forest = BipartiteRandomForestRegressor(
        n_estimators=5,
        min_rows_leaf=5,
        min_cols_leaf=7,
        bipartite_adapter="gmosa",
        criterion="squared_error_gso",
        random_state=random_state,
    )
    model = BipartiteModelForestRegressor(forest, leaf_estimator)
    model.fit(XX, Y)
    return model


def test_tree_predict(data, fitted_tree):
    XX, Y = data
    y_hat = fitted_tree.predict(XX)
    assert y_hat.size == Y.size
    assert mean_squared_error(Y.ravel(), y_hat) < y_hat.var() / 2


def test_forest_predict(data, fitted_forest):
    XX, Y = data
    y_hat = fitted_forest.predict(XX)
    assert y_hat.size == Y.size
    assert mean_squared_error(Y.ravel(), y_hat) < y_hat.var() / 2