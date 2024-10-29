
from pathlib import Path
import sys

import pytest
from sklearn.metrics import mean_squared_error, roc_auc_score
from bipartite_learn.datasets import NuclearReceptorsLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from dwnn import KroneckerWeightedNeighbors


@pytest.fixture()
def data():
    return NuclearReceptorsLoader().load()


def test_dwnn(data):
    XX, Y = data
    model = KroneckerWeightedNeighbors(metric="precomputed", weights="similarity")
    model.fit(XX, Y)
    Y_hat = model.predict([XX[0][:23], XX[1][:43]]).reshape((23, 43))
    score = roc_auc_score(Y[:23, :43].ravel(), Y_hat.ravel())
    assert score > 0.9