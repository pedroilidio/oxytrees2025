from types import MappingProxyType

DEFAULT_PARAMS = MappingProxyType(
    dict(
        bipartite_adapter="gmosa",
        n_estimators=200,
        max_samples=None,
        bootstrap=False,
        random_state=0,
        verbose=10,
    )
)


def small_bxt_gmo():
    import bipartite_learn.ensemble

    return bipartite_learn.ensemble.BipartiteExtraTreesRegressor(
        n_estimators=5,
        max_depth=3,
        random_state=0,
        verbose=0,
    )


def bxt_gmo():
    import bipartite_learn.ensemble

    return bipartite_learn.ensemble.BipartiteExtraTreesRegressor(
        criterion="squared_error",
        **DEFAULT_PARAMS,
    )


def bxt_bgso():
    import bipartite_learn.ensemble

    return bipartite_learn.ensemble.BipartiteExtraTreesRegressor(
        criterion="squared_error_gso",
        **DEFAULT_PARAMS,
    )
