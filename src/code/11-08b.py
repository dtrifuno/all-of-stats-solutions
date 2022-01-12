import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

bs = [0.25, 0.5, 1, 5]
ns = [3, 100, 500, 10000]

fig, axs = plt.subplots(2, 2, figsize=(14, 12))

for i, n in enumerate(ns):
    ax = axs[i // 2, i % 2]
    ax.set(xlabel="Sample average", ylabel="Probability", title=f"n = {n}")
    xs = np.linspace(-10 / np.sqrt(n), 10 / np.sqrt(n), 500)
    wald_p = 2 * stats.norm.cdf(-np.sqrt(n) * np.abs(xs))
    ax.plot(xs, wald_p, label="Wald p-value")
    for b in bs:
        t = np.sqrt(b ** 2 * n + 1)
        bayes_p = t / (
            t + np.exp(xs ** 2 * b ** 2 * n ** 2 / (2 * (b ** 2 * n + 1)))
        )
        ax.plot(
            xs, bayes_p, label=f"P(H_0|x), b={b}", lw=1.2, linestyle="dashed"
        )
    ax.legend()

fig.savefig("11-08b.png", bbox_inches="tight")
