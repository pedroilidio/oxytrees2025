import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

###
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = 9
###

markers = ["x", "t", "o", "v"]
linestyle = ["-", "--", "-.", ":"]

data = pd.read_csv("fit_time_bxt_bdt.csv")
data["log_n_samples"] = np.log10(data.n)
data["log_time"] = np.log10(data.time)

# Store the slope estimates and their standard errors
slope_estimates = {}
standard_errors = {}
n = {}

# Compute linear regression coefficients and their uncertainty
plt.figure(figsize=(4, 7))
for i, (estimator, subset) in enumerate(data.groupby("estimator")):
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        subset["log_n_samples"], subset["log_time"]
    )
    print(f"{estimator}: {slope=:.4f}, {intercept=:.4f}, {std_err=:.2g}")
    slope_estimates[estimator] = slope
    standard_errors[estimator] = std_err
    n[estimator] = subset.shape[0]

    # Plot the results
    sns.regplot(
        data=subset,
        y="log_time",
        x="log_n_samples",
        truncate=False,
        marker=".",
        line_kws={
           #"linestyle": linestyle[i],
        },
        scatter_kws={
            #"s": 5,
            # "edgecolor": "white",
            # "zorder": 10,
        },
        # facet_kws={"legend_out": False},
        label=f"{estimator} ({slope = :.4f} $\\pm$ {std_err:.2g})",
    )

plt.legend()
plt.tight_layout()
plt.savefig("time_vs_n_artificial_data_loglog.png")
plt.savefig("time_vs_n_artificial_data_loglog.pdf", bbox_inches="tight", transparent=True)
plt.clf()

plt.figure(figsize=(4, 7))
for estimator1, estimator2 in (
    ("bxt_gso", "bxt_gmo"),
    ("bdt_gso", "bdt_gmo"),
    ("bdt_gso", "bxt_gmo"),
):
    # Calculate the test statistic
    b1 = slope_estimates[estimator1]
    b2 = slope_estimates[estimator2]
    SE1 = standard_errors[estimator1]
    SE2 = standard_errors[estimator2]
    test_statistic = (b1 - b2) / ((SE1**2 + SE2**2) ** 0.5)

    # Calculate the degrees of freedom
    df = n[estimator1] + n[estimator2] - 4

    # Calculate the p-value
    p_value = 2 * stats.t.sf(abs(test_statistic), df)

    print(f"({estimator1} vs. {estimator2}) {p_value=:.2g}")

sns.scatterplot(
    data=data,
    y="time",
    x="n",
    hue="estimator",
    #markers={
    #    "bxt_gso": "t",
    #    "bxt_gmo": markers[1],
    #    "bdt_gso": markers[2],
    #    "bdt_gmo": markers[3],
    #}
)

plt.tight_layout()
plt.savefig("time_vs_n_artificial_data.png")
plt.savefig("time_vs_n_artificial_data.pdf", bbox_inches="tight", transparent=True)
print(f"Saved figure to time_vs_n_artificial_data.pdf")
plt.clf()

# Output:
# bdt_gmo: slope=3.2558, intercept=-6.6471, std_err=0.0042
# bdt_gso: slope=2.9904, intercept=-6.5577, std_err=0.0017
# bxt_gmo: slope=3.0919, intercept=-6.9531, std_err=0.0077
# bxt_gso: slope=2.9779, intercept=-6.7731, std_err=0.0027
# (bxt_gso vs. bxt_gmo) p_value=6e-23
# (bdt_gso vs. bdt_gmo) p_value=2.5e-65
# (bdt_gso vs. bxt_gmo) p_value=5.2e-21
