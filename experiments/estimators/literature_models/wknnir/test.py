import sys
from pathlib import Path

import pytest
import numpy as np
from bipartite_learn.model_selection import (
    MultipartiteGridSearchCV,
    make_multipartite_kfold,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wknnir import WkNNIR


@pytest.fixture
def random_state():
    return 0


@pytest.fixture
def data(random_state):
    rng = np.random.default_rng(random_state)
    return {
        "X": [rng.random((17, 17)), rng.random((19, 19))],
        "y": rng.integers(0, 2, size=(17, 19)).astype("float64"),
        "pairwise": True,
    }


def test_wknnir(random_state, data):
    estimator = WkNNIR(k=5, kr=5, T=0.7)
    estimator.fit(data["X"], data["y"])
    pred = estimator.predict([data["X"][0][:3, :], data["X"][1][:2, :]])
    assert pred.shape == (6,)


def test_wknnir_grid(random_state, data):
    estimator = MultipartiteGridSearchCV(
        WkNNIR(k=7, kr=7, T=0.8),
        param_grid={"k": [1, 2, 3, 5, 7, 9], "kr": [1], "T": np.arange(0.1, 1.1, 0.1)},
        cv=make_multipartite_kfold(
            n_parts=2,  # Bipartite
            cv=2,
            shuffle=True,
            diagonal=False,
            random_state=random_state,
        ),
        n_jobs=1,
        scoring="average_precision",
        pairwise=True,
        error_score="raise",
    )
    estimator.fit(data["X"], data["y"])
    pred = estimator.predict([data["X"][0][:3, :], data["X"][1][:2, :]])
    assert np.isnan(pred).sum() == 0
    assert pred.shape == (6,)
