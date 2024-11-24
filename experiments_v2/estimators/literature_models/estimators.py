import numpy as np

from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor
from imblearn.under_sampling import RandomUnderSampler
from bipartite_learn.pipeline import make_multipartite_pipeline
from bipartite_learn.preprocessing.multipartite import DTHybridSampler
from bipartite_learn.preprocessing.monopartite import (
    TargetKernelLinearCombiner,
    TargetKernelDiffuser,
    SimilarityDistanceSwitcher,
    SymmetryEnforcer,
)
from bipartite_learn.neighbors import WeightedNeighborsRegressor
from bipartite_learn.wrappers import LocalMultiOutputWrapper, GlobalSingleOutputWrapper
from bipartite_learn.matrix_factorization import (
    NRLMFClassifier,
    DNILMFClassifier,
)
from bipartite_learn.model_selection import (
    MultipartiteGridSearchCV,
    MultipartiteRandomizedSearchCV,
    make_multipartite_kfold,
)

from .kron_rls import KronRLSRegressor

RSTATE = 0
N_JOBS = 1

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

kfold = make_multipartite_kfold(
    n_parts=2,  # Bipartite
    cv=2,
    shuffle=True,
    diagonal=False,
    random_state=RSTATE,
)


blmnii_rls = MultipartiteGridSearchCV(
    make_multipartite_pipeline(
        TargetKernelLinearCombiner(),
        LocalMultiOutputWrapper(
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
        ),
    ),
    param_grid={
        "targetkernellinearcombiner__samplers__alpha": ALPHA_OPTIONS,
    },
    cv=kfold,
    n_jobs=N_JOBS,
    scoring="average_precision",
    pairwise=True,
)

blmnii_svm = MultipartiteGridSearchCV(
    make_multipartite_pipeline(
        TargetKernelLinearCombiner(),
        LocalMultiOutputWrapper(
            primary_rows_estimator=WeightedNeighborsRegressor(
                metric="precomputed",
                weights="similarity",
            ),
            primary_cols_estimator=WeightedNeighborsRegressor(
                metric="precomputed",
                weights="similarity",
            ),
            secondary_rows_estimator=MultiOutputRegressor(SVR(kernel="precomputed")),
            secondary_cols_estimator=MultiOutputRegressor(SVR(kernel="precomputed")),
            independent_labels=True,
        ),
    ),
    param_grid={
        "targetkernellinearcombiner__samplers__alpha": ALPHA_OPTIONS,
    },
    cv=kfold,
    n_jobs=N_JOBS,
    scoring="average_precision",
    pairwise=True,
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

# van Laarhoven
lmo_rls = MultipartiteGridSearchCV(
    make_multipartite_pipeline(
        TargetKernelLinearCombiner(),
        LocalMultiOutputWrapper(
            primary_rows_estimator=KernelRidge(kernel="precomputed"),
            primary_cols_estimator=KernelRidge(kernel="precomputed"),
            secondary_rows_estimator=KernelRidge(kernel="precomputed"),
            secondary_cols_estimator=KernelRidge(kernel="precomputed"),
            independent_labels=False,
        ),
    ),
    param_grid={
        "targetkernellinearcombiner__samplers__alpha": ALPHA_OPTIONS,
    },
    cv=kfold,
    n_jobs=N_JOBS,
    scoring="average_precision",
    pairwise=True,
)

kron_rls = MultipartiteGridSearchCV(
    make_multipartite_pipeline(
        TargetKernelLinearCombiner(),
        KronRLSRegressor(),
    ),
    param_grid={
        "targetkernellinearcombiner__samplers__alpha": ALPHA_OPTIONS,
    },
    cv=kfold,
    n_jobs=N_JOBS,
    scoring="average_precision",
    pairwise=True,
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
    cv=kfold,
    n_jobs=N_JOBS,
    scoring="average_precision",
    pairwise=True,
)
