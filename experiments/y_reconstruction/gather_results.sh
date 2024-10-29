#!/bin/bash
set -e

python make_runs_table.py --out y_reconstruction/results.tsv --runs y_reconstruction/runs bipartite_adaptations/runs

# Generate y_reconstruction/results_renamed.tsv
python y_reconstruction/rename_estimators.py y_reconstruction/results.tsv
