from bipartite_learn.ensemble import BipartiteExtraTreesRegressor

RANDOM_STATE = 0
COMMON_PARAMS = dict(
    bipartite_adapter="gmosa",
    n_estimators=1000,
    max_samples=None,
    bootstrap=False,
    random_state=RANDOM_STATE,
    verbose=10,
)

small_bxt_gmo = BipartiteExtraTreesRegressor(
    n_estimators=5,
    max_depth=3,
    random_state=RANDOM_STATE,
    verbose=0,
)

bxt_gmo = BipartiteExtraTreesRegressor(
    criterion="squared_error",
    **COMMON_PARAMS,
)

bxt_bgso = BipartiteExtraTreesRegressor(
    criterion="squared_error_gso",
    **COMMON_PARAMS,
)