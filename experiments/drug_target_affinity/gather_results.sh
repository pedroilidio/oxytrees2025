#!/bin/bash
set -e

# This script's parent directory
BASEDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Generate results_renamed.tsv
python make_runs_table.py \
    --out $BASEDIR/results.tsv \
    --runs $BASEDIR/runs

python $BASEDIR/rename_estimators.py $BASEDIR/results.tsv

