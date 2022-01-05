import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)

b = 10000
n = 50

data = rng.uniform(0, 1, size=n)
est_theta = data.max()

bootstrap_data = rng.choice(data, size=(n, b))
nonparam_bootstrap_thetas = bootstrap_data.max(axis=0)

bootstrap_data = rng.uniform(0, est_theta, size=(n, b))
param_bootstrap_thetas = bootstrap_data.max(axis=0)

xmin = 0.8
xmax = 1
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))

ax1.hist(nonparam_bootstrap_thetas, bins=120, density=True, range=(xmin, xmax))
ax1.set(
    xlabel="Theta-hat",
    ylabel="Density",
    title="Nonparametric bootstrap distribution",
)
ax2.hist(param_bootstrap_thetas, bins="auto", density=True, range=(xmin, xmax))
ax2.set(
    xlabel="Theta-hat",
    ylabel="Density",
    title="Parametric bootstrap distribution",
)

xs = np.linspace(xmin, xmax, 100)
ys = 50 * xs ** 49
ax3.plot(xs, ys)
ax3.set(xlabel="Theta-hat", ylabel="Density", title="True distribution")

plt.savefig("09-10.png", bbox_inches="tight")
