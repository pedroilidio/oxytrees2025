import argparse
from pathlib import Path
from time import perf_counter
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from bipartite_learn import tree
from sklearn.utils import check_random_state
# from make_examples import make_interaction_regression


def collect_run_times(outdir=Path.cwd(), n_jobs=1, random_state=None, cache_dir=None):
    random_state = check_random_state(random_state)
    memory = joblib.Memory(location=cache_dir, verbose=0)

    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    outpath = outdir / f"{datetime.now()}.csv"

    estimators = {
        # "bxt_gso": tree.BipartiteExtraTreeRegressor(
        #     random_state=random_state,
        #     criterion="squared_error_gso",
        #     bipartite_adapter="gmosa",
        # ),
        # "bdt_gso": tree.BipartiteDecisionTreeRegressor(
        #     random_state=random_state,
        #     criterion="squared_error_gso",
        #     bipartite_adapter="gmosa",
        # ),
        # "bxt_gmo": tree.BipartiteExtraTreeRegressor(
        #     random_state=random_state,
        #     criterion="squared_error",
        #     bipartite_adapter="gmosa",
        # ),
        # "bdt_gmo": tree.BipartiteDecisionTreeRegressor(
        #     random_state=random_state,
        #     criterion="squared_error",
        #     bipartite_adapter="gmosa",
        # ),
        # "ss_bdt_gmo_mse": tree.BipartiteDecisionTreeRegressorSS(
        #     random_state=random_state,
        #     criterion="squared_error",
        #     bipartite_adapter="gmosa",
        #     supervision=0.5,
        #     unsupervised_criterion_rows="squared_error",
        #     unsupervised_criterion_cols="squared_error",
        # ),
        # "ss_bdt_gmo_ad": tree.BipartiteDecisionTreeRegressorSS(
        #     random_state=random_state,
        #     criterion="squared_error",
        #     bipartite_adapter="gmosa",
        #     supervision=0.5,
        #     unsupervised_criterion_rows="squared_error",
        #     unsupervised_criterion_cols="squared_error",
        #     axis_decision_only=True,
        # ),
        # "ss_bdt_gmo_md": tree.BipartiteDecisionTreeRegressorSS(
        #     random_state=random_state,
        #     criterion="squared_error",
        #     bipartite_adapter="gmosa",
        #     supervision=0.5,
        #     unsupervised_criterion_rows="mean_distance",
        #     unsupervised_criterion_cols="mean_distance",
        #     axis_decision_only=True,
        # ),
        "ss_bdt_gso_mse": tree.BipartiteDecisionTreeRegressorSS(
            random_state=random_state,
            criterion="squared_error_gso",
            bipartite_adapter="gmosa",
            supervision=0.5,
            unsupervised_criterion_rows="squared_error",
            unsupervised_criterion_cols="squared_error",
        ),
        "ss_bdt_gso_ad": tree.BipartiteDecisionTreeRegressorSS(
            random_state=random_state,
            criterion="squared_error_gso",
            bipartite_adapter="gmosa",
            supervision=0.5,
            unsupervised_criterion_rows="squared_error",
            unsupervised_criterion_cols="squared_error",
            axis_decision_only=True,
        ),
        "ss_bdt_gso_md": tree.BipartiteDecisionTreeRegressorSS(
            random_state=random_state,
            criterion="squared_error_gso",
            bipartite_adapter="gmosa",
            supervision=0.5,
            unsupervised_criterion_rows="mean_distance",
            unsupervised_criterion_cols="mean_distance",
            axis_decision_only=True,
        ),
    }

    def make_data(n):
        print(f"Making data for {n=}...")
        *X, y = random_state.random((3, n, n))
        print(f"Made data for {n=}.")
        return X, y

    def run_estimator(estimator_name, n):
        print(f"Getting data for {estimator_name=}, {n=}...")
        X, y = cached_make_data(n)
        estimator = estimators[estimator_name]

        print(f"Fitting {estimator_name=}, {n=}...")
        t0 = perf_counter()
        estimator.fit(X, y)
        record = {
            "time": perf_counter() - t0,
            "n": n,
            "estimator": estimator_name,
        }
        print(f"Fitted: {estimator_name=}, {n=} time={record['time']:.2f} s.")

        return record

    cached_make_data = memory.cache(make_data)
    cached_run_estimator = memory.cache(run_estimator)

    try:
        # TODO: one process for each estimator.
        records = []
        for record in joblib.Parallel(
            n_jobs=n_jobs,
            return_as="generator_unordered",  # Enable recovering partial results
            prefer="processes",
        )(
            joblib.delayed(cached_run_estimator)(estimator_name, n)
            for n in np.logspace(2, 4, 50, dtype=int)
            for estimator_name in estimators.keys()
        ):
            records.append(record)
    finally:
        pd.DataFrame.from_records(records).to_csv(outpath, index=False)
        print(f"Saved {len(records)} records to {outpath}.")


def main():
    parser = argparse.ArgumentParser(
        description="Collect run times for different estimators on artificial data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--outdir",
        default="./results",
        help="Path to save results.",
    )
    parser.add_argument(
        "--n-jobs",
        default=1,
        type=int,
        help="Number of jobs.",
    )
    parser.add_argument(
        "--cache-dir",
        default="./cache",
        help="Cache directory to store checkpoints.",
    )
    parser.add_argument(
        "--random-state",
        default=0,
        type=int,
        help="Random state.",
    )
    args = parser.parse_args()

    collect_run_times(
        outdir=args.outdir,
        n_jobs=args.n_jobs,
        cache_dir=args.cache_dir,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
