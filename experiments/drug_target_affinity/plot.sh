#!/bin/bash
set -e

# This script's parent directory
BASEDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

ESTIMATORS=(
    "bxt_gso"
    "bxt_gmosa"
    "moltrans"
    "deep_dta"
    "kron_rls"
)

python make_statistical_comparisons.py \
    --results-table $BASEDIR/results_renamed.tsv \
    --outdir $BASEDIR/statistical_comparisons \
    --estimators ${ESTIMATORS[@]}

