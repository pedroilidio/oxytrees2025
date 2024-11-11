from bipartite_learn.ensemble import BipartiteExtraTreesRegressor


bxt_gmo = BipartiteExtraTreesRegressor(
    n_estimators=10,
    max_depth=5,
    n_jobs=5,
    random_state=0,
    verbose=50,
)
