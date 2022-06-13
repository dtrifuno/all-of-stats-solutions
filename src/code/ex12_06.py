import matplotlib.pyplot as plt
import numpy as np


def james_stein_estimator(data):
    k = data.size
    return np.max((1 - (k - 2) / np.sum(data ** 2)), 0) * data


rng = np.random.default_rng(42)
sims = 25_000
ks = [3, 10, 50, 500]

fig1, axs1 = plt.subplots(2, 2, figsize=(14, 12))
fig2, axs2 = plt.subplots(2, 2, figsize=(14, 12))

for theta_fn, axs in ((np.ones, axs1), (np.arange, axs2)):
    for i, k in enumerate(ks):
        ax = axs[i // 2, i % 2]
        theta = theta_fn(k)
        data = rng.normal(theta, size=(sims, k))
        js_estimates = np.apply_along_axis(james_stein_estimator, 0, data)
        js_errors = ((js_estimates - theta) ** 2).mean(axis=1)
        mle_errors = ((data - theta) ** 2).mean(axis=1)
        ax.set(xlabel="Mean squared error", ylabel="Density")
        ax.set(title=f"k = {k}")
        ax.hist(
            js_errors,
            bins="auto",
            density=True,
            histtype="step",
            label="James-Stein",
        )
        ax.hist(
            mle_errors, bins="auto", density=True, histtype="step", label="MLE"
        )
        ax.legend()

fig1.savefig("12-06a.png", bbox_inches="tight")
fig2.savefig("12-06b.png", bbox_inches="tight")
