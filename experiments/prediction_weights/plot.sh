#!/bin/bash
set -e

BASEDIR=prediction_weights
RENAMING=$BASEDIR/estimator_renaming.yml
METRICS=(
    "fit_time"
    "score_time"
    "fit_score_time"
    "LT+TL_average_precision"
    "LT+TL_roc_auc"
    "TT_average_precision"
    "TT_roc_auc"
)
BXT_ESTIMATORS=(
    "bxt_gmosa"
    "bxt_gmo__softmax"
    "bxt_gmo__precomputed"
    "bxt_gmo__uniform"
    "bxt_gmo__square"
    "bxt_gmo_full"
)
BRF_ESTIMATORS=(
    "brf_gmosa"
    "brf_gmo__softmax"
    "brf_gmo__precomputed"
    "brf_gmo__uniform"
    "brf_gmo__square"
    "brf_gmo_full"
)

python make_statistical_comparisons.py \
    --results-table $BASEDIR/results.tsv \
    --outdir $BASEDIR/statistical_comparisons/bxt \
    --estimators ${BXT_ESTIMATORS[@]} \
    --metrics ${METRICS[@]} \
    --estimator-renaming $RENAMING

python make_statistical_comparisons.py \
    --results-table $BASEDIR/results.tsv \
    --outdir $BASEDIR/statistical_comparisons/brf \
    --estimators ${BRF_ESTIMATORS[@]} \
    --metrics ${METRICS[@]} \
    --estimator-renaming $RENAMING
