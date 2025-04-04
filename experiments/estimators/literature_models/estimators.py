import numpy as np

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from imblearn.under_sampling import RandomUnderSampler
from bipartite_learn.pipeline import make_multipartite_pipeline
from bipartite_learn.preprocessing.multipartite import DTHybridSampler
from bipartite_learn.preprocessing.monopartite import TargetKernelLinearCombiner
from bipartite_learn.neighbors import WeightedNeighborsRegressor
from bipartite_learn.wrappers import LocalMultiOutputWrapper, GlobalSingleOutputWrapper
from bipartite_learn.model_selection import (
    MultipartiteGridSearchCV,
    make_multipartite_kfold,
)

from .kron_rls import KronRLSRegressor
from .wknnir import WkNNIR

# Controls the balance between the similarity kernel and network-based kernel.
ALPHA_OPTIONS = [
    0.0,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    1.0,
]
GRID_SEARCH_PARAMS = dict(
    cv=make_multipartite_kfold(
        n_parts=2,  # Bipartite
        cv=2,
        shuffle=True,
        diagonal=False,
        random_state=0,
    ),
    n_jobs=1,
    scoring="average_precision",
    pairwise=True,
    verbose=2,
)

blmnii_rls_core = LocalMultiOutputWrapper(
    primary_rows_estimator=WeightedNeighborsRegressor(
        metric="precomputed",
        weights="similarity",
    ),
    primary_cols_estimator=WeightedNeighborsRegressor(
        metric="precomputed",
        weights="similarity",
    ),
    secondary_rows_estimator=KernelRidge(kernel="precomputed"),
    secondary_cols_estimator=KernelRidge(kernel="precomputed"),
    independent_labels=False,
)

blmnii_rls = MultipartiteGridSearchCV(
    make_multipartite_pipeline(
        TargetKernelLinearCombiner(),
        blmnii_rls_core,
    ),
    param_grid={"targetkernellinearcombiner__samplers__alpha": ALPHA_OPTIONS},
    **GRID_SEARCH_PARAMS,
)

blmnii_svm = clone(blmnii_rls).set_params(
    estimator__localmultioutputwrapper__secondary_rows_estimator=MultiOutputRegressor(
        SVR(kernel="precomputed")
    ),
    estimator__localmultioutputwrapper__secondary_cols_estimator=MultiOutputRegressor(
        SVR(kernel="precomputed")
    ),
    estimator__localmultioutputwrapper__independent_labels=True,
)

dthybrid_regressor = make_multipartite_pipeline(
    DTHybridSampler(),
    LocalMultiOutputWrapper(
        primary_rows_estimator=WeightedNeighborsRegressor(
            metric="precomputed",
            weights="similarity",
        ),
        primary_cols_estimator=WeightedNeighborsRegressor(
            metric="precomputed",
            weights="similarity",
        ),
        secondary_rows_estimator=WeightedNeighborsRegressor(
            metric="precomputed",
            weights="similarity",
        ),
        secondary_cols_estimator=WeightedNeighborsRegressor(
            metric="precomputed",
            weights="similarity",
        ),
        independent_labels=True,
    ),
)

# RLS-avg [Van Laarhoven, 2011]
lmo_rls_core = LocalMultiOutputWrapper(
    primary_rows_estimator=KernelRidge(kernel="precomputed"),
    primary_cols_estimator=KernelRidge(kernel="precomputed"),
    secondary_rows_estimator=KernelRidge(kernel="precomputed"),
    secondary_cols_estimator=KernelRidge(kernel="precomputed"),
    independent_labels=False,
)

lmo_rls = MultipartiteGridSearchCV(
    make_multipartite_pipeline(
        TargetKernelLinearCombiner(),
        lmo_rls_core,
    ),
    param_grid={"targetkernellinearcombiner__samplers__alpha": ALPHA_OPTIONS},
    **GRID_SEARCH_PARAMS,
)

# RLS-Kron [Van Laarhoven, 2011]
kron_rls = MultipartiteGridSearchCV(
    make_multipartite_pipeline(
        TargetKernelLinearCombiner(),
        KronRLSRegressor(),
    ),
    param_grid={"targetkernellinearcombiner__samplers__alpha": ALPHA_OPTIONS},
    **GRID_SEARCH_PARAMS,
)

mlp = MultipartiteGridSearchCV(
    GlobalSingleOutputWrapper(
        estimator=MLPRegressor(),
        under_sampler=RandomUnderSampler(),
    ),
    param_grid={
        "estimator__hidden_layer_sizes": [
            (100,) * 5,
            (100,) * 10,
            (200, 100, 100, 100, 50),
            (1024, 512, 256, 128, 64, 32),
        ],
    },
    **GRID_SEARCH_PARAMS,
)

logistic = GlobalSingleOutputWrapper(LogisticRegression())

wknnir = MultipartiteGridSearchCV(
    WkNNIR(k=7, kr=7, T=0.8),
    param_grid={"k": [1, 2, 3, 5, 7, 9], "kr": [1, 7], "T": np.arange(0.1, 1.1, 0.1)},
    **GRID_SEARCH_PARAMS,
)
