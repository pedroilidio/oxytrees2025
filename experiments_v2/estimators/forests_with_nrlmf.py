from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
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

# "sqrt" as used by the original code source:
#
# Pliakos K, Vens C. Drug-target interaction prediction with tree-ensemble learning
# and output space reconstruction. BMC Bioinformatics. 2020;21:1–11. 

nrlmf__bxt_gmo = make_multipartite_pipeline(
    nrlmf_sampler,
    clone(bxt_gmo).set_params(max_row_features="sqrt", max_col_features="sqrt"),
)
nrlmf__bxt_bgso = make_multipartite_pipeline(
    nrlmf_sampler,
    clone(bxt_bgso).set_params(max_row_features="sqrt", max_col_features="sqrt"),
)

nrlmf__dwnn_similarities__bxt_bgso = make_multipartite_pipeline(
    nrlmf_sampler,
    clone(dwnn_similarities_bxt_bgso),
    memory=memory,
)
nrlmf__dwnn_square__bxt_bgso = make_multipartite_pipeline(
    nrlmf_sampler,
    FunctionTransformer(np.square),
    clone(dwnn_similarities_bxt_bgso),
    memory=memory,
)

nrlmf__bxt_bgso_kronrls = make_multipartite_pipeline(
    nrlmf_sampler,
    clone(bxt_bgso_kronrls),
    memory=memory,
)

nrlmf__uniform__bxt_bgso = make_multipartite_pipeline(
    nrlmf_sampler,
    clone(uniform_bxt_bgso),
    memory=memory,
)
