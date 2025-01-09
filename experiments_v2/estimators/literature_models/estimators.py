import numpy as np

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import make_pipeline
from bipartite_learn.pipeline import make_multipartite_pipeline
from bipartite_learn.preprocessing.multipartite import DTHybridSampler
# from bipartite_learn.preprocessing.monopartite import (
#     TargetKernelLinearCombiner,
#     TargetKernelDiffuser,
# )
from bipartite_learn.neighbors import WeightedNeighborsRegressor
from bipartite_learn.wrappers import LocalMultiOutputWrapper, GlobalSingleOutputWrapper
from bipartite_learn.model_selection import (
    MultipartiteGridSearchCV,
    make_multipartite_kfold,
)

from .kron_rls import KronRLSRegressor
from .wknnir import WkNNIR

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

blmnii_rls = LocalMultiOutputWrapper(
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

# Below is the version using network kernel as the original paper. Since the network
# kernel is only used during fitting, the primary estimators would receive very
# different similarity metrics at prediction time, and thus we choose not to use it.

# blmnii_rls = MultipartiteGridSearchCV(
#     LocalMultiOutputWrapper(
#         primary_rows_estimator=make_pipeline(
#             TargetKernelLinearCombiner(),
#             WeightedNeighborsRegressor(
#                 metric="precomputed",
#                 weights="similarity",
#             ),
#         ),
#         primary_cols_estimator=make_pipeline(
#             TargetKernelLinearCombiner(),
#             WeightedNeighborsRegressor(
#                 metric="precomputed",
#                 weights="similarity",
#             ),
#         ),
#         secondary_rows_estimator=KernelRidge(kernel="precomputed"),
#         secondary_cols_estimator=KernelRidge(kernel="precomputed"),
#         independent_labels=False,
#     ),
#     param_grid=[
#         {
#             "primary_rows_estimator__targetkernellinearcombiner__alpha": alpha,
#             "primary_cols_estimator__targetkernellinearcombiner__alpha": alpha,
#         }
#         for alpha in ALPHA_OPTIONS
#     ],
#     cv=kfold,
#     n_jobs=N_JOBS,
#     scoring="average_precision",
#     pairwise=True,
# )

blmnii_svm = clone(blmnii_rls).set_params(
    secondary_rows_estimator=MultiOutputRegressor(SVR(kernel="precomputed")),
    secondary_cols_estimator=MultiOutputRegressor(SVR(kernel="precomputed")),
    independent_labels=True,
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
# Only the similarity kernel is used, not the network kernel. See the explanation above
# for blmnii_rls.
lmo_rls = LocalMultiOutputWrapper(
    primary_rows_estimator=KernelRidge(kernel="precomputed"),
    primary_cols_estimator=KernelRidge(kernel="precomputed"),
    secondary_rows_estimator=KernelRidge(kernel="precomputed"),
    secondary_cols_estimator=KernelRidge(kernel="precomputed"),
    independent_labels=False,
)

# RLS-Kron [Van Laarhoven, 2011]
# Only the similarity kernel is used, not the network kernel. See the explanation above
# for blmnii_rls.
kron_rls = KronRLSRegressor()

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

logistic = GlobalSingleOutputWrapper(LogisticRegression())

wknnir = MultipartiteGridSearchCV(
    WkNNIR(k=7, kr=7, T=0.8)
    param_grid={"k": [1, 2, 3, 5, 7, 9], "kr": [1], "T": np.arange(0.1, 1.1, 0.1)},
    cv=kfold,
    n_jobs=N_JOBS,
    scoring="average_precision",
    pairwise=True,
)
