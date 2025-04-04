from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import FunctionTransformer
from bipartite_learn.pipeline import make_multipartite_pipeline

from ..literature_models.nrlmf import nrlmf_sampler
from ..model_forests.bipartite_model_trees import BipartiteModelForestRegressor
from ..model_forests.dwnn import KroneckerWeightedNeighbors
from .estimators import ss_bxt_gso__md_size

CACHE_DIR = Path(__file__).parent.resolve() / "cache"
# memory = joblib.Memory(location=CACHE_DIR, verbose=0)
memory = str(CACHE_DIR)

COMMON_PARAMS = dict(
    bipartite_adapter="gmosa",
    criterion="squared_error_gso",
    n_estimators=200,
    random_state=0,
    verbose=10,
)

dwnn_similarities__ss_md_size__bxt_bgso = BipartiteModelForestRegressor(
    estimator=clone(ss_bxt_gso__md_size).set_params(min_rows_leaf=5, min_cols_leaf=5),
    leaf_estimator=KroneckerWeightedNeighbors(
        metric="precomputed", weights="similarity"
    ),
    pairwise=True,
)

dwnn_square__ss_md_size__bxt_bgso = make_multipartite_pipeline(
    # NOTE: One could also apply the function as the `weights` parameter in the
    # leaf_estimator, but since trees are invariant to monotonic transformations,
    # it is more efficient to apply the transformation here.
    FunctionTransformer(np.square),
    clone(dwnn_similarities__ss_md_size__bxt_bgso),
    memory=memory,
)

nrlmf__dwnn_square__ss_md_size__bxt_bgso = make_multipartite_pipeline(
    nrlmf_sampler,
    FunctionTransformer(np.square),
    clone(dwnn_similarities__ss_md_size__bxt_bgso),
    memory=memory,
)
