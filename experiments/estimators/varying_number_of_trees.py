from types import MappingProxyType

from .bipartite_forests import bxt_bgso
from .model_forests.estimators import bxt_bgso_kronrls
from .forests_with_nrlmf import nrlmf__bxt_gmo


DEFAULT_FOREST_PARAMS = MappingProxyType(
    dict(
        bipartite_adapter="gmosa",
        # n_estimators=500,
        n_estimators=200,
        bootstrap=False,
        max_samples=None,
        random_state=0,
        verbose=10,
        warm_start=True,  # Important in this context
    )
)


def oxytrees():
    estimator = bxt_bgso_kronrls()
    # Also set warm_start=True to the wrapper estimator
    estimator.set_params(warm_start=True)
    forest = estimator.estimator
    forest.set_params(**DEFAULT_FOREST_PARAMS)
    forest.set_params(
        min_cols_leaf=5,
        min_rows_leaf=5,
        max_row_features=None,
        max_col_features=None,
    )
    assert estimator.estimator.min_cols_leaf == 5
    assert estimator.estimator.min_rows_leaf == 5
    return estimator


def oxytrees_noleaf():
    return oxytrees().estimator


def bictr():
    estimator = nrlmf__bxt_gmo()
    forest = estimator.named_steps["bipartiteextratreesregressor"]
    forest.set_params(**DEFAULT_FOREST_PARAMS)
    forest.set_params(
        max_row_features="sqrt",
        max_col_features="sqrt",
        min_cols_leaf=1,
        min_rows_leaf=1,
    )
    # FIXME: We are trusting pipeline caching to not train transformers again.
    forest2 = estimator.named_steps["bipartiteextratreesregressor"]
    assert forest2.n_estimators == DEFAULT_FOREST_PARAMS["n_estimators"]
    return estimator
