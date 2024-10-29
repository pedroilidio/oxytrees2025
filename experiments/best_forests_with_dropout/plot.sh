#!/bin/bash
set -e

DEBUG=$1

BASEDIR=best_forests_with_dropout
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
ESTIMATORS=(
    # "nrlmf"
    "bxt_gso"
    # "bxt_gmo"
    # "bxt_gmosa"
    # "bxt_gso_1k"
    # "bxt_gmosa_1k"
    "bxt_gso__nrlmf"
    "bxt_gmosa__nrlmf"
    "bxt_gmo__nrlmf"
    "brf_gmo__nrlmf"
    "brf_lmo"
    "md_size"
    "md_fixed"
    "ad_size"
    "ad_fixed"
    "mse_density"
)

# gmosa_1k
# brf_gso_1k__nrlmf\
# bxt_gso_1k__nrlmf\

echo "*** NO DROP ***"
python $DEBUG make_statistical_comparisons.py \
    --results-table $BASEDIR/results_renamed.tsv \
    --outdir $BASEDIR/statistical_comparisons/no_drop \
    --estimators ${ESTIMATORS[@]} \
    --metrics ${METRICS[@]} \
    --estimator-renaming $RENAMING \
    --raise-missing

for drop in 50 70 90; do
    echo "*** DROP $drop% ***"
    python $DEBUG make_statistical_comparisons.py \
        --results-table $BASEDIR/results_renamed.tsv \
        --outdir $BASEDIR/statistical_comparisons/drop$drop \
        --estimators $(for E in ${ESTIMATORS[@]}; do echo $E"__"$drop; done) \
        --metrics ${METRICS[@]} \
        --estimator-renaming $RENAMING \
        --raise-missing
done