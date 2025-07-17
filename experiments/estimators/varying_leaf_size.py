from types import MappingProxyType

from .bipartite_forests import bxt_bgso
from .model_forests.estimators import bxt_bgso_kronrls
from .forests_with_nrlmf import nrlmf__bxt_gmo


# Values chosen: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 50, 100]


DEFAULT_FOREST_PARAMS = MappingProxyType(
    dict(
        bipartite_adapter="gmosa",
        n_estimators=50,
        max_row_features="sqrt",
        max_col_features="sqrt",
        bootstrap=False,
        max_samples=None,
        random_state=0,
        verbose=10,
    )
)


def oxytrees_n(n):
    estimator = bxt_bgso_kronrls()
    forest = estimator.estimator
    forest.set_params(**DEFAULT_FOREST_PARAMS)
    forest.set_params(min_cols_leaf=n, min_rows_leaf=n)
    assert estimator.estimator.min_cols_leaf == n
    assert estimator.estimator.min_rows_leaf == n
    return estimator


def bictr_n(n):
    estimator = nrlmf__bxt_gmo()
    forest = estimator.named_steps["bipartiteextratreesregressor"]
    forest.set_params(**DEFAULT_FOREST_PARAMS)
    forest.set_params(min_cols_leaf=n, min_rows_leaf=n)

    forest2 = estimator.named_steps["bipartiteextratreesregressor"]
    assert forest2.min_cols_leaf == n
    assert forest2.min_rows_leaf == n

    return estimator


def bgso_n(n):
    forest = bxt_bgso()
    forest.set_params(**DEFAULT_FOREST_PARAMS)
    forest.set_params(min_cols_leaf=n, min_rows_leaf=n)
    assert forest.min_cols_leaf == n
    assert forest.min_rows_leaf == n
    return forest


# Oxytrees


def oxytrees_2():
    return oxytrees_n(2)


def oxytrees_3():
    return oxytrees_n(3)


def oxytrees_4():
    return oxytrees_n(4)


def oxytrees_5():
    return oxytrees_n(5)


def oxytrees_6():
    return oxytrees_n(6)


def oxytrees_7():
    return oxytrees_n(7)


def oxytrees_8():
    return oxytrees_n(8)


def oxytrees_9():
    return oxytrees_n(9)


def oxytrees_10():
    return oxytrees_n(10)


def oxytrees_11():
    return oxytrees_n(11)


def oxytrees_12():
    return oxytrees_n(12)


def oxytrees_13():
    return oxytrees_n(13)


def oxytrees_14():
    return oxytrees_n(14)


def oxytrees_15():
    return oxytrees_n(15)


def oxytrees_20():
    return oxytrees_n(20)


def oxytrees_50():
    return oxytrees_n(50)


def oxytrees_100():
    return oxytrees_n(100)


# BICTR


def bictr_2():
    return bictr_n(2)


def bictr_3():
    return bictr_n(3)


def bictr_4():
    return bictr_n(4)


def bictr_5():
    return bictr_n(5)


def bictr_6():
    return bictr_n(6)


def bictr_7():
    return bictr_n(7)


def bictr_8():
    return bictr_n(8)


def bictr_9():
    return bictr_n(9)


def bictr_10():
    return bictr_n(10)


def bictr_11():
    return bictr_n(11)


def bictr_12():
    return bictr_n(12)


def bictr_13():
    return bictr_n(13)


def bictr_14():
    return bictr_n(14)


def bictr_15():
    return bictr_n(15)


def bictr_20():
    return bictr_n(20)


def bictr_50():
    return bictr_n(50)


def bictr_100():
    return bictr_n(100)


# BGSO


def bgso_2():
    return bgso_n(2)


def bgso_3():
    return bgso_n(3)


def bgso_4():
    return bgso_n(4)


def bgso_5():
    return bgso_n(5)


def bgso_6():
    return bgso_n(6)


def bgso_7():
    return bgso_n(7)


def bgso_8():
    return bgso_n(8)


def bgso_9():
    return bgso_n(9)


def bgso_10():
    return bgso_n(10)


def bgso_11():
    return bgso_n(11)


def bgso_12():
    return bgso_n(12)


def bgso_13():
    return bgso_n(13)


def bgso_14():
    return bgso_n(14)


def bgso_15():
    return bgso_n(15)


def bgso_20():
    return bgso_n(20)


def bgso_50():
    return bgso_n(50)


def bgso_100():
    return bgso_n(100)
