#!/bin/bash
set -e

BASEDIR=bipartite_adaptations
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

python make_statistical_comparisons.py \
    --results-table $BASEDIR/results.tsv \
    --outdir $BASEDIR/statistical_comparisons/brf \
    --estimators \
        brf_gmo brf_gmosa brf_lmo brf_lso brf_gso brf_sgso_us \
    --metrics ${METRICS[@]} \
    --estimator-renaming $RENAMING \
    --raise-missing

python make_statistical_comparisons.py \
    --results-table $BASEDIR/results.tsv \
    --outdir $BASEDIR/statistical_comparisons/bxt \
    --estimators \
        bxt_gmo bxt_gmosa bxt_lmo bxt_lso bxt_gso bxt_sgso_us \
    --metrics ${METRICS[@]} \
    --estimator-renaming $RENAMING \
    --raise-missing

python make_statistical_comparisons.py \
    --results-table $BASEDIR/results.tsv \
    --outdir $BASEDIR/statistical_comparisons/gso \
    --estimators \
        bxt_gso bxt_sgso bxt_sgso_us \
        bxt_gso bxt_sgso bxt_sgso_us \
    --metrics ${METRICS[@]} \
    --estimator-renaming $RENAMING \
    --raise-missing
