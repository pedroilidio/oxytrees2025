import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
from tqdm import trange
import matplotlib.pyplot as plt


def rank_correlation_single_metric(
    n_runs: int = 10_000,
    n_samples: int = 100,
    out: Path = Path("rank_correlation_aupr.png"),
    n_densities: int = 10,
    random_state: int = 0,
    metric: str = "aupr",  # "aupr" or "auroc"
    fig_size: list[int] = (12, 6),
):
    assert metric in ("aupr", "auroc")
    metric_func = average_precision_score if metric == "aupr" else roc_auc_score

    rng = np.random.default_rng(random_state)
    ranks = np.linspace(0, 1, n_samples)

    plt.figure(figsize=fig_size)

    # Iter over label frequencies ([1:] excludes 0.0)
    for iter, density in enumerate(
        np.linspace(0, 1, n_densities + 1, endpoint=False)[1:]
    ):
        tag = f"[density {iter}/{n_densities} ({density:.2g})]"

        print(tag, "Making noisy labels (step 1/3)...")
        # Each row of y is a run of ranked labels
        y = rng.choice(2, (n_runs, n_samples), p=(1 - density, density))
        
        # Compute the AUC and AP for each run
        ap = np.empty(n_runs, dtype=float)

        print(tag, "Computing metric (step 2/3)...")
        for i in trange(n_runs):
            ap[i] = metric_func(y[i], ranks)
            # ap[i] = average_precision_score(y[i], ranks)

        ap_corr = np.empty(n_samples, dtype=float)

        # Compute the correlation between ranks and scores
        print(tag, "Computing correlations (step 3/3)...")
        for r in trange(n_samples):
            ap_corr[r], _ = stats.pointbiserialr(y[:, r], ap)
        
        # Plot
        plt.plot(ranks, ap_corr, label=f"{density:.2f}")

    plt.legend()
    plt.xlabel("Rank")
    plt.ylabel("Correlation")
    plt.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)

    print("Done. Figure saved to", out)


def rank_correlation(
    out: Path,
    n_runs: int,
    n_samples: int,
    density: float = 0.5,
    fig_size: list[int] = (12, 6),
    random_state: int = 0
):
    assert 0 < density < 1.0

    rng = np.random.default_rng(random_state)

    # Each row of y is a run of ranked labels
    print("Making noisy labels (step 1/3)...")
    y = rng.choice(2, (n_runs, n_samples), p=(1 - density, density))
    
    # Compute the AUC and AP for each run
    ranks = np.linspace(0, 1, n_samples)
    auc = np.empty(n_runs, dtype=float)
    ap = np.empty(n_runs, dtype=float)

    print("Computing AUC and AP (step 2/3)...")
    for i in trange(n_runs):
        auc[i] = roc_auc_score(y[i], ranks)
        ap[i] = average_precision_score(y[i], ranks)

    auc_corr = np.empty(n_samples, dtype=float)
    ap_corr = np.empty(n_samples, dtype=float)

    # Compute the correlation between ranks and scores
    print("Computing correlations (step 3/3)...")
    for r in trange(n_samples):
        auc_corr[r], _ = stats.pointbiserialr(y[:, r], auc)
        ap_corr[r], _ = stats.pointbiserialr(y[:, r], ap)
    
    print("Making figure...")
    plt.figure(figsize=fig_size)
    plt.plot(ranks, auc_corr, label="AUROC")
    plt.plot(ranks, ap_corr, label="AUPR")
    plt.axvline(
        (1 - density),
        color="black",
        linestyle="--",
        label="Ideal threshold (1 - density)",
    )
    plt.legend()
    plt.xlabel("Rank")
    plt.ylabel("Correlation")
    plt.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)

    print("Done. Figure saved to", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("results/rank_correlation.pdf")
    )
    parser.add_argument("--n_runs", type=int, default=100_000)
    parser.add_argument("--fig_size", type=int, nargs=2, default=[12, 6])
    parser.add_argument("--n_samples", type=int, default=1_000)
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--n_densities", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=0)
    args = parser.parse_args()

    rank_correlation(
        out=args.out,
        n_runs=args.n_runs,
        n_samples=args.n_samples,
        density=args.density,
        fig_size=args.fig_size,
        random_state=args.random_state
    )
    for metric in ("aupr", "auroc"):
        rank_correlation_single_metric(
            metric=metric,
            n_runs=args.n_runs // args.n_densities,
            n_samples=args.n_samples,
            out=args.out.with_stem(args.out.stem + "_" + metric),
            fig_size=args.fig_size,
            n_densities=args.n_densities,
            random_state=args.random_state,
        )


if __name__ == "__main__":
    main()