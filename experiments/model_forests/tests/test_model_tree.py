from pathlib import Path
import sys

import pytest
from sklearn.datasets import load_iris
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))
from model_trees import ModelTree, ModelForestRegressor

@pytest.fixture
def data():
    iris = load_iris()
    y = OneHotEncoder(sparse=False).fit_transform(iris.target.reshape(-1, 1))
    X = iris.data
    return X, y


@pytest.mark.parametrize("single_output", [True, False])
@pytest.mark.parametrize("pairwise", [True, False])
def test_model_tree(data, single_output, pairwise):
    X, y = data
    if single_output:
        y = y[:, 1].ravel()
    if pairwise:
        X = rbf_kernel(X)
    tree = DecisionTreeRegressor(min_samples_leaf=10)
    leaf = KNeighborsRegressor()
    model = ModelTree(tree, leaf, pairwise=pairwise)
    model.fit(X, y)
    y_hat = model.predict(X)
    assert y_hat.shape == y.shape


def test_model_forest(data):
    X, y = data
    forest = RandomForestRegressor(min_samples_leaf=10)
    leaf = KNeighborsRegressor()
    model = ModelForestRegressor(forest, leaf)
    model.fit(X, y)
    y_hat = model.predict(X)
    assert y_hat.shape == y.shape