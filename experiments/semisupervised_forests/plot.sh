#!/bin/bash
set -e

DEBUG=$1

BASEDIR=semisupervised_forests
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
    "ad_fixed"
    "ad_density"
    "ad_size"
    "ad_random"
    "md_fixed"
    "md_density"
    "md_size"
    "md_random"
    "mse_fixed"
    "mse_density"
    "mse_size"
    "mse_random"
)

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
