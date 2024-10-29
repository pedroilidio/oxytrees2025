#!/bin/bash
set -e

DEBUG=$1
echo "DEBUG=$DEBUG"
BASEDIR=literature_models
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
    "nrlmf"
    # "dnilmf"  # XXX
    "lmo_rls"
    "kron_rls"
    "mlp"
    "blmnii_rls"
    "blmnii_svm"
    "dthybrid"
    # "bxt_sgso_us"
    # "bxt_gso_1k"
    # "bxt_gmosa_1k"
    "bxt_gmosa__nrlmf"
    "bxt_gso__nrlmf"
    "bxt_gmo__nrlmf"
    # "brf_lmo"
    # "ss_bxt_gso__md_size"
    # "ss_bxt_gso__ad_fixed"
    # "ss_bxt_gso__mse_density"
)

echo "*** NO DROP ***"
python $DEBUG make_statistical_comparisons.py \
    --results-table $BASEDIR/results_renamed.tsv \
    --outdir $BASEDIR/statistical_comparisons/no_drop \
    --estimators ${ESTIMATORS[@]} \
    --metrics ${METRICS[@]} \
    --raise-missing \
    --estimator-renaming $RENAMING \


for drop in 50 70 90; do
    echo "*** DROP $drop% ***"
    python $DEBUG make_statistical_comparisons.py \
        --results-table $BASEDIR/results_renamed.tsv \
        --outdir $BASEDIR/statistical_comparisons/drop$drop \
        --estimators $(for E in ${ESTIMATORS[@]}; do echo $E"__"$drop; done) \
        --metrics ${METRICS[@]} \
        --raise-missing \
    --estimator-renaming $RENAMING \

done