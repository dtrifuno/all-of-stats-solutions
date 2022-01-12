import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

data = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0])
n = data.size
k = data.sum()

fig1, axs1 = plt.subplots(2, 2, figsize=(13, 12))
fig2, axs2 = plt.subplots(2, 2, figsize=(13, 12))

xs = np.linspace(0.001, 0.999, 100)
likelihood = stats.binom.pmf(k, n, xs)

for i, alpha in enumerate((0.5, 1, 10, 100)):
    prior = stats.beta.pdf(xs, alpha, alpha)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    ax1 = axs1[i // 2, i % 2]
    ax2 = axs2[i // 2, i % 2]

    ax1.set(xlabel="p", ylabel="Density", title=f"Beta({alpha}, {alpha}) prior")
    ax1.plot(xs, prior)
    ax2.set(
        xlabel="p",
        ylabel="Posterior density",
        title=f"Beta({alpha}, {alpha}) prior",
    )
    ax2.plot(xs, posterior)

fig1.savefig("11-05a.png", bbox_inches="tight")
fig2.savefig("11-05b.png", bbox_inches="tight")
