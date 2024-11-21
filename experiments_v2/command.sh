python scripts/run_experiments.py \
    --estimator-definitions config/estimators.yml \
    --dataset-definitions config/datasets.yml \
    --fold-definitions config/fold_definitions.yml \
    --experiment-definitions config/experiments.yml \
    --scoring-definitions config/scoring.yml \
    --output-directory mlruns \
    --code-path ./
