import numpy as np
import bipartite_learn.ensemble
from sklearn.utils import check_random_state

import bipartite_positive_dropper
from wrappers import regressor_to_binary_classifier


COMMON_PARAMS = dict(
    bipartite_adapter='gmosa',
    criterion='squared_error_gso',
    n_estimators=100,
    n_jobs=4,
    random_state=0,
    verbose=10,
)


# Not currently being used
def positive_dropper_regressor_to_classifier(estimator, drop):
    return regressor_to_binary_classifier(
        bipartite_positive_dropper.wrap_estimator(estimator),
        drop=drop,
        random_state=COMMON_PARAMS["random_state"],
    )


# def density_updater(
#     y,
#     sample_indices,
#     col_indices,
#     start,
#     end,
#     start_col,
#     end_col,
#     **_,
# ):
#     node_indices = np.ix_(
#         sample_indices[start:end],
#         col_indices[start_col:end_col],
#     )
#     return 0.1 + 0.9 * np.array(y)[node_indices].mean()


def density_updater(node_value, **_):
    # node_value contains the mean of the node's y partition, which is the
    # density for binary labels.
    return 0.1 + 0.9 * node_value


def node_size_updater(weighted_n_samples, weighted_n_node_samples, **_):
    return 1.0 - weighted_n_node_samples / weighted_n_samples


def random_updater(**_):
    return random_updater._rng.random()
random_updater._rng = check_random_state(COMMON_PARAMS["random_state"])


# Basic semi-supervised estimators
# ================================

ss_bxt_gso__mse_fixed = bipartite_learn.ensemble.BipartiteExtraTreesRegressorSS(
    supervision=0.5,
    **COMMON_PARAMS,
)
ss_bxt_gso__mse_density = bipartite_learn.ensemble.BipartiteExtraTreesRegressorSS(
    update_supervision=density_updater,
    **COMMON_PARAMS,
)
ss_bxt_gso__mse_size = bipartite_learn.ensemble.BipartiteExtraTreesRegressorSS(
    update_supervision=node_size_updater,
    **COMMON_PARAMS,
)
ss_bxt_gso__mse_random = bipartite_learn.ensemble.BipartiteExtraTreesRegressorSS(
    update_supervision=random_updater,
    **COMMON_PARAMS,
)


# Axis decision only
# ==================

ss_bxt_gso__ad_fixed = bipartite_learn.ensemble.BipartiteExtraTreesRegressorSS(
    axis_decision_only=True,
    supervision=0.0,  # In this case setting to 0 makes sense.
    **COMMON_PARAMS,
)
ss_bxt_gso__ad_density = bipartite_learn.ensemble.BipartiteExtraTreesRegressorSS(
    axis_decision_only=True,
    update_supervision=density_updater,
    **COMMON_PARAMS,
)
ss_bxt_gso__ad_size = bipartite_learn.ensemble.BipartiteExtraTreesRegressorSS(
    axis_decision_only=True,
    update_supervision=node_size_updater,
    **COMMON_PARAMS,
)
ss_bxt_gso__ad_random = bipartite_learn.ensemble.BipartiteExtraTreesRegressorSS(
    axis_decision_only=True,
    update_supervision=random_updater,
    **COMMON_PARAMS,
)


# Mean distance unsupervised impurity
# ===================================

ss_bxt_gso__md_fixed = bipartite_learn.ensemble.BipartiteExtraTreesRegressorSS(
    unsupervised_criterion_rows='mean_distance',
    unsupervised_criterion_cols='mean_distance',
    supervision=0.5,
    **COMMON_PARAMS,
)
ss_bxt_gso__md_density = bipartite_learn.ensemble.BipartiteExtraTreesRegressorSS(
    unsupervised_criterion_rows='mean_distance',
    unsupervised_criterion_cols='mean_distance',
    update_supervision=density_updater,
    **COMMON_PARAMS,
)
ss_bxt_gso__md_size = bipartite_learn.ensemble.BipartiteExtraTreesRegressorSS(
    unsupervised_criterion_rows='mean_distance',
    unsupervised_criterion_cols='mean_distance',
    update_supervision=node_size_updater,
    **COMMON_PARAMS,
)
ss_bxt_gso__md_random = bipartite_learn.ensemble.BipartiteExtraTreesRegressorSS(
    unsupervised_criterion_rows='mean_distance',
    unsupervised_criterion_cols='mean_distance',
    update_supervision=random_updater,
    **COMMON_PARAMS,
)
