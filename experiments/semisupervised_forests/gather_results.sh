python make_runs_table.py \
    --out semisupervised_forests/results.tsv \
    --runs semisupervised_forests/runs \
&& python semisupervised_forests/rename_estimators.py semisupervised_forests/results.tsv
