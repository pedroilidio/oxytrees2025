import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

SEED = 0
N = 1000

rng = np.random.default_rng(SEED)
y_true = rng.integers(2, size=N)
y_pred = 0.5 * y_true + rng.random(N)
# y_pred = y_true

order = np.argsort(y_pred)

p = y_true.mean()
n = 1 - p

c = y_true[order]
C = np.cumsum(c) / N
r = np.arange(1, N + 1) / N
# r = np.arange(N) / N
# r = np.linspace(0, 1, N)
# r = np.linspace(0, 1, N, endpoint=False)

mpr = np.mean(c * r)
mpr_max = np.mean(np.sort(y_true) * r)
mpr_min = np.mean(np.sort(y_true)[::-1] * r)
mpr_max_man = 0.5 - n ** 2 / 2
mpr_min_man = p ** 2 / 2

print("p:", p, "n:", n)

print(
    "AUROC:",
    f"{roc_auc_score(y_true, y_pred)} (sklearn)",
    f"{0.5 + 1/n * (0.5 - np.mean(C) / p)} (theoretical)",
    f"{0.5 + 1/n * (np.mean(c * r) / p - 0.5)} (theoretical on MPR)",
    sep="\n  * ",
)
print(
    "MPR:",
    mpr,
    f"{(mpr - mpr_min) / (mpr_max - mpr_min)} (normalized)",
    f"{1 / n * (mpr/p - .5) + .5} (normalized on theoretical min/max)",
    sep="\n  * ",
)

mask = r != 1
Cpr = (1 - C[mask]/p) / (1 - r[mask])
Cprc = Cpr * c[mask]

print(
    "AP:",
    average_precision_score(y_true, y_pred),
    np.mean(Cprc),
    (p / 2) * (1 + np.mean(Cpr ** 2)),
    sep="\n  * ",
)
