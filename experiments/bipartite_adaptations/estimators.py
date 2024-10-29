from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from imblearn.under_sampling import RandomUnderSampler

from bipartite_learn.wrappers import (
    GlobalSingleOutputWrapper,
    LocalMultiOutputWrapper,
)
from bipartite_learn.ensemble import (
    BipartiteRandomForestRegressor,
    BipartiteExtraTreesRegressor,
)

__all__ = [
    "bxt_lmo",
    "brf_lmo",
    "bxt_lso",
    "brf_lso",
    "bxt_sgso",
    "brf_sgso",
    "bxt_sgso_us",
    "brf_sgso_us",
    "bxt_gso",
    "brf_gso",
    "bxt_gmo",
    "brf_gmo",
    "bxt_gmosa",
    "brf_gmosa",
]

COMMON_PARAMS = dict(
    random_state=0,
    n_estimators=100,
    max_depth=None,
    n_jobs=3,
    verbose=10,
    # max_samples=0.7,
    # bootstrap=True,  # Default is True for RF and False for ET
)

GLOBAL_FOREST_PARAMS = dict(
    **COMMON_PARAMS,
)
LOCAL_FOREST_PARAMS = COMMON_PARAMS | dict(
    criterion="squared_error",
    n_estimators=COMMON_PARAMS["n_estimators"] // 2,  # Half for each axis
)

# Setting a constant fraction max_features=a makes bipartite and non-bipartite
# forests comparable, since the number of features per node will be the same:
#
#   a * (n_row_features + n_col_features) = a * n_row_features + a * n_col_features
#
# "sqrt" or "log" as max_features are not linear functions and thus do not work
# in the same way.
RF_PARAMS = dict(
    max_features=0.5,
)
BRF_PARAMS = dict(
    max_row_features=0.5,
    max_col_features=0.5,
)


# Multi-output forests for LMO wrapper
_local_xt = ExtraTreesRegressor(
    **LOCAL_FOREST_PARAMS,
)

_local_rf = RandomForestRegressor(
    **LOCAL_FOREST_PARAMS,
    **RF_PARAMS,
)

# Compositions of single output forests for LMO wrapper
_local_so_xt = MultiOutputRegressor(
    clone(_local_xt).set_params(n_jobs=1, verbose=1),
    n_jobs=COMMON_PARAMS["n_jobs"],
)

_local_so_rf = MultiOutputRegressor(
    clone(_local_rf).set_params(n_jobs=1, verbose=1),
    n_jobs=COMMON_PARAMS["n_jobs"],
)

bxt_lmo = LocalMultiOutputWrapper(
    primary_rows_estimator=_local_xt,
    primary_cols_estimator=_local_xt,
    secondary_rows_estimator=_local_xt,
    secondary_cols_estimator=_local_xt,
    independent_labels=False,
)

brf_lmo = LocalMultiOutputWrapper(
    primary_rows_estimator=_local_rf,
    primary_cols_estimator=_local_rf,
    secondary_rows_estimator=_local_rf,
    secondary_cols_estimator=_local_rf,
    independent_labels=False,
)

bxt_lso = LocalMultiOutputWrapper(
    primary_rows_estimator=_local_so_xt,
    primary_cols_estimator=_local_so_xt,
    secondary_rows_estimator=_local_so_xt,
    secondary_cols_estimator=_local_so_xt,
    independent_labels=True,
)

brf_lso = LocalMultiOutputWrapper(
    primary_rows_estimator=_local_so_rf,
    primary_cols_estimator=_local_so_rf,
    secondary_rows_estimator=_local_so_rf,
    secondary_cols_estimator=_local_so_rf,
    independent_labels=True,
)

# Naive global single output forests
bxt_sgso = GlobalSingleOutputWrapper(
    ExtraTreesRegressor(
        criterion="squared_error",
        **GLOBAL_FOREST_PARAMS,
    )
)

brf_sgso = GlobalSingleOutputWrapper(
    RandomForestRegressor(
        criterion="squared_error",
        **RF_PARAMS,
        **GLOBAL_FOREST_PARAMS,
    )
)

# Global single output forests with under-sampling
bxt_sgso_us = GlobalSingleOutputWrapper(
    ExtraTreesRegressor(
        criterion="squared_error",
        **GLOBAL_FOREST_PARAMS,
    ),
    under_sampler=RandomUnderSampler(),
)

brf_sgso_us = GlobalSingleOutputWrapper(
    RandomForestRegressor(
        criterion="squared_error",
        **RF_PARAMS,
        **GLOBAL_FOREST_PARAMS,
    ),
    under_sampler=RandomUnderSampler(),
)

# Global multi-output forests
bxt_gmo = BipartiteExtraTreesRegressor(
    criterion="squared_error",
    bipartite_adapter="gmo",
    prediction_weights="square",
    min_rows_leaf=5,
    min_cols_leaf=5,
    **GLOBAL_FOREST_PARAMS,
)


brf_gmo = BipartiteRandomForestRegressor(
    criterion="squared_error",
    bipartite_adapter="gmo",
    prediction_weights="square",
    min_rows_leaf=5,
    min_cols_leaf=5,
    **BRF_PARAMS,
    **GLOBAL_FOREST_PARAMS,
)


# Global multi-output forests with single label average
bxt_gmosa = BipartiteExtraTreesRegressor(
    criterion="squared_error",
    bipartite_adapter="gmosa",
    **GLOBAL_FOREST_PARAMS,
)

brf_gmosa = BipartiteRandomForestRegressor(
    criterion="squared_error",
    bipartite_adapter="gmosa",
    **BRF_PARAMS,
    **GLOBAL_FOREST_PARAMS,
)

# Global single output forests
bxt_gso = BipartiteExtraTreesRegressor(
    criterion="squared_error_gso",
    bipartite_adapter="gmosa",
    **GLOBAL_FOREST_PARAMS,
)

brf_gso = BipartiteRandomForestRegressor(
    criterion="squared_error_gso",
    bipartite_adapter="gmosa",
    **BRF_PARAMS,
    **GLOBAL_FOREST_PARAMS,
)
