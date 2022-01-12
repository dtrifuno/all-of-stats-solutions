import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

bs = [0.25, 0.5, 1, 5]
xs = np.linspace(-15, 15, 1000)

fig, ax = plt.subplots(1, figsize=(14, 12))
ax.set(xlabel="x", ylabel="Probability")

wald_p = 2 * stats.norm.cdf(-np.abs(xs))
ax.plot(xs, wald_p, label="Wald p-value")
for b in bs:
    t = np.sqrt(b ** 2 + 1)
    bayes_p = t / (t + np.exp(xs ** 2 * b ** 2 / (2 * b ** 2 + 2)))
    ax.plot(xs, bayes_p, label=f"P(H_0|x), b={b}", lw=1.2, linestyle="dashed")

ax.legend()
fig.savefig("11-08a.png", bbox_inches="tight")
