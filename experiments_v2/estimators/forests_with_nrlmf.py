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
    bxt_bgso_kronrls,
)

CACHE_DIR = Path(__file__).parent.resolve() / "cache"
# memory = joblib.Memory(location=CACHE_DIR, verbose=0)
memory = str(CACHE_DIR)

nrlmf__bxt_bgso = make_multipartite_pipeline(nrlmf_sampler, bxt_bgso)
nrlmf__bxt_gmo = make_multipartite_pipeline(nrlmf_sampler, bxt_gmo)

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
