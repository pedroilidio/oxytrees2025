import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    make_scorer,
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from imblearn.under_sampling import RandomUnderSampler
import bipartite_learn.ensemble
from bipartite_learn.base import BaseBipartiteEstimator
from bipartite_learn.preprocessing.monopartite import SymmetryEnforcer
from bipartite_learn.pipeline import make_multipartite_pipeline
import DeepPurpose.DTI
import DeepPurpose.utils

from drug_target_affinity.deep_purpose_wrapper import DeepPurposeWrapper


def restrict_scorer_to_known_outputs(scorer_func):
    """Use only known affinities to score.

    Unknown affinities are set to a very high value, which becomes very low
    after -log(y_true) is applied in the preprocessing step. We then assume an
    unknown drug-target pair is always present in the training set, so that
    this defining value to exclude can be selected with y_true.min(). Even if
    it is not the case, the damage is expected be fairly small, since the
    y_true.min() cutout will likely drop one or very few known interactions in
    this case.
    """ 
    def new_func(y_true, y_pred):
        minimum = y_true.min()
        # If there are NaN, NaN will be the minimum.
        if minimum == np.nan:
            mask = np.isfinite(y_true)
        else:
            mask = y_true > minimum
        return scorer_func(y_true[mask], y_pred[mask])
    return new_func
    

def exclude_min_binarizer(y):
    """Mark known affinities so that we can undersample the unknown ones.

    Used in conjunction with DeepPurposeWrapper.
    """
    return (y > y.min()).astype(int)


def load_kiba_affinities_for_deepdta(path, max_value=20, fill_nan=False):
    data = max_value - pd.read_table(path, index_col=0)
    if fill_nan:
        return data.fillna(value=0).values
    return data.values


def get_scorers():
    """Used in yaml config files to get our custom scorers."""
    return SCORERS


FOREST_PARAMS = {
    "bipartite_adapter": "gmosa",
    "n_estimators": 1000,
    "n_jobs": 20,
    "verbose": 10,
}
SCORERS = {
    "explained_variance": "explained_variance",
    "max_error": "max_error",
    "neg_mean_absolute_error": "neg_mean_absolute_error",
    "neg_mean_squared_error": "neg_mean_squared_error",
    "neg_root_mean_squared_error": "neg_root_mean_squared_error",
    "neg_median_absolute_error": "neg_median_absolute_error",
    "r2": "r2",
    "explained_variance_known": make_scorer(
        restrict_scorer_to_known_outputs(explained_variance_score),
        greater_is_better=True,
    ),
    "max_error_known": make_scorer(
        restrict_scorer_to_known_outputs(max_error),
        greater_is_better=False,
    ),
    "neg_mean_absolute_error_known": make_scorer(
        restrict_scorer_to_known_outputs(mean_absolute_error),
        greater_is_better=False,
    ),
    "neg_mean_squared_error_known": make_scorer(
        restrict_scorer_to_known_outputs(mean_squared_error),
        greater_is_better=False,
    ),
    "neg_median_absolute_error_known": make_scorer(
        restrict_scorer_to_known_outputs(median_absolute_error),
        greater_is_better=False,
    ),
    "r2_known": make_scorer(
        restrict_scorer_to_known_outputs(r2_score),
        greater_is_better=True,
    ),
}


print(f"[{__file__}] {torch.cuda.is_available()=}")
print(f"[{__file__}] {torch.cuda.device_count()=}")
if torch.cuda.is_available():
    print(f"[{__file__}] {torch.cuda.get_device_name(0)=}")

# ========================================================
# Enable usage of tensor cores for matmul and convolutions
# ========================================================
# (from https://pytorch.org/docs/stable/notes/cuda.html)

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
print(f"[{__file__}] {torch.backends.cuda.matmul.allow_tf32=}")
print(f"[{__file__}] {torch.backends.cudnn.allow_tf32=}")


# ===============================
# Deep learning models definition
# ===============================

# NOTE: DeepPuccose selects GPU automatically if available.
deep_dta = DeepPurposeWrapper(
    DeepPurpose.utils.generate_config(
        drug_encoding="CNN",
        target_encoding="CNN",
        cls_hidden_dims=[1024, 1024, 512],
        train_epoch=100,
        LR=0.001,
        batch_size=256,
        cnn_drug_filters=[32, 64, 96],
        cnn_target_filters=[32, 64, 96],
        cnn_drug_kernels=[4, 6, 8],
        cnn_target_kernels=[4, 8, 12],
        cuda_id=0,
    ),
    # Selected balanced number of unknown interactions:
    # under_sampler=RandomUnderSampler(random_state=0),
    # binarizer=exclude_min_binarizer,
)

# MolTrans hyperparameters are obtained from
# https://github.com/kexinhuang12345/MolTrans/blob/master/config.py
#
#    config['batch_size'] = 16
#    config['input_dim_drug'] = 23532
#    config['input_dim_target'] = 16693
#    config['train_epoch'] = 13
#    config['max_drug_seq'] = 50
#    config['max_protein_seq'] = 545
#    config['emb_size'] = 384
#    config['dropout_rate'] = 0.1
#
#    # DenseNet
#    config['scale_down_ratio'] = 0.25
#    config['growth_rate'] = 20
#    config['transition_rate'] = 0.5
#    config['num_dense_blocks'] = 4
#    config['kernal_dense_size'] = 3
#
#    # Encoder
#    config['intermediate_size'] = 1536
#    config['num_attention_heads'] = 12
#    config['attention_probs_dropout_prob'] = 0.1
#    config['hidden_dropout_prob'] = 0.1
#    config['flat_dim'] = 78192

moltrans = DeepPurposeWrapper(
    DeepPurpose.utils.generate_config(
        drug_encoding="Transformer",
        target_encoding="Transformer",
        input_dim_drug=23532,
        input_dim_protein=16693,
        train_epoch=13,
        batch_size=16,
        transformer_dropout_rate=0.1,
        transformer_emb_size_drug=384,
        transformer_intermediate_size_drug=1536,
        transformer_num_attention_heads_drug=12,
        transformer_emb_size_target=384,
        transformer_intermediate_size_target=1536,
        transformer_num_attention_heads_target=12,
        transformer_attention_probs_dropout=0.1,
        transformer_hidden_dropout_rate=0.1,
        cls_hidden_dims=[1024, 1024, 512],
        LR=0.001,
    ),
    # Selected balanced number of unknown interactions:
    # under_sampler=RandomUnderSampler(random_state=0),
    # binarizer=exclude_min_binarizer,
)


# ============================
# Bipartite forests definition
# ============================

brf_gmosa = bipartite_learn.ensemble.BipartiteRandomForestRegressor(
    criterion="squared_error",
    max_row_features=0.5,
    max_col_features=0.5,
    **FOREST_PARAMS,
)

brf_gso = bipartite_learn.ensemble.BipartiteRandomForestRegressor(
    criterion="squared_error_gso",
    max_row_features=0.5,
    max_col_features=0.5,
    **FOREST_PARAMS,
)

bxt_gmosa = bipartite_learn.ensemble.BipartiteExtraTreesRegressor(
    criterion="squared_error",
    **FOREST_PARAMS,
)

bxt_gso = bipartite_learn.ensemble.BipartiteExtraTreesRegressor(
    criterion="squared_error_gso",
    **FOREST_PARAMS,
)

bgbm = bipartite_learn.ensemble.BipartiteGradientBoostingRegressor(
    criterion="friedman_gso",
    **{k: v for k, v in FOREST_PARAMS.items() if k != "n_jobs"},
)

