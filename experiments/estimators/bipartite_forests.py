RANDOM_STATE = 0
COMMON_PARAMS = dict(
    bipartite_adapter="gmosa",
    n_estimators=200,
    max_samples=None,
    bootstrap=False,
    random_state=RANDOM_STATE,
    verbose=10,
)


def small_bxt_gmo():
    import bipartite_learn.ensemble
    return bipartite_learn.ensemble.BipartiteExtraTreesRegressor(
        n_estimators=5,
        max_depth=3,
        random_state=RANDOM_STATE,
        verbose=0,
    )


def bxt_gmo():
    import bipartite_learn.ensemble
    return bipartite_learn.ensemble.BipartiteExtraTreesRegressor(
        criterion="squared_error",
        **COMMON_PARAMS,
    )


def bxt_bgso():
    import bipartite_learn.ensemble
    return bipartite_learn.ensemble.BipartiteExtraTreesRegressor(
        criterion="squared_error_gso",
        **COMMON_PARAMS,
    )
