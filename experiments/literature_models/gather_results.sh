#!/bin/bash
set -e

BASEDIR=literature_models
# Generate results_renamed.tsv
python make_runs_table.py \
    --out $BASEDIR/results.tsv \
    --runs \
        bipartite_adaptations/runs \
        y_reconstruction/runs \
        best_forests_with_dropout/runs \
        $BASEDIR/runs

python best_forests_with_dropout/rename_estimators.py $BASEDIR/results.tsv

