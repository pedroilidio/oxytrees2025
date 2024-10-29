#!/bin/bash
set -e

BASEDIR=best_forests_with_dropout

python make_runs_table.py \
    --out $BASEDIR/results.tsv \
    --runs \
        bipartite_adaptations/runs \
        y_reconstruction/runs \
        semisupervised_forests/runs \
        $BASEDIR/runs

# Generate results_renamed.tsv
python $BASEDIR/rename_estimators.py $BASEDIR/results.tsv
