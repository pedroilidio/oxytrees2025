from pathlib import Path

import numpy as np
from sklearn.preprocessing import FunctionTransformer
from bipartite_learn.pipeline import make_multipartite_pipeline

from .bipartite_forests import bxt_bgso, bxt_gmo
from .literature_models.nrlmf import nrlmf_sampler
from .model_forests.estimators import (
    dwnn_similarities_bxt_bgso,
    uniform_bxt_bgso,
    bxt_bgso_kronrls,
)

CACHE_DIR = Path(__file__).parent.resolve() / "cache"
# memory = joblib.Memory(location=CACHE_DIR, verbose=0)
memory = str(CACHE_DIR)


def nrlmf__bxt_gmo():
    return make_multipartite_pipeline(
        nrlmf_sampler(),
        bxt_gmo().set_params(max_row_features="sqrt", max_col_features="sqrt"),
        # "sqrt" as used by the original code source:
        #     Pliakos K, Vens C. Drug-target interaction prediction with tree-ensemble
        #     learning and output space reconstruction. BMC Bioinformatics. 2020;21:1–11.
    )


def nrlmf__bxt_bgso():
    return make_multipartite_pipeline(
        nrlmf_sampler(),
        bxt_bgso().set_params(max_row_features="sqrt", max_col_features="sqrt"),
    )


def nrlmf__dwnn_similarities__bxt_bgso():
    return make_multipartite_pipeline(
        nrlmf_sampler(),
        dwnn_similarities_bxt_bgso(),
        memory=memory,
    )


def nrlmf__dwnn_square__bxt_bgso():
    return make_multipartite_pipeline(
        nrlmf_sampler(),
        FunctionTransformer(np.square),
        dwnn_similarities_bxt_bgso(),
        memory=memory,
    )


def nrlmf__bxt_bgso_kronrls():
    return make_multipartite_pipeline(
        nrlmf_sampler(),
        bxt_bgso_kronrls(),
        memory=memory,
    )


def nrlmf__uniform__bxt_bgso():
    return make_multipartite_pipeline(
        nrlmf_sampler(),
        uniform_bxt_bgso(),
        memory=memory,
    )
