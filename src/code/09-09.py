import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

rng = np.random.default_rng(42)
np.set_printoptions(precision=2)

b = 20_000
n = 100
xlimits = (100, 210)
alpha = 0.05

fig, axs = plt.subplots(2, 2, figsize=(14, 11))

# true distribution
xs = rng.normal(5, size=(n, b))
true_theta_dist = np.exp(xs.mean(axis=0))
axs[0, 0].hist(true_theta_dist, density=True, range=xlimits, bins=70)
axs[0, 0].set(xlabel="Theta-hat", ylabel="Density", title="True distribution")

data = rng.normal(5, size=n)
xbar = data.mean()
est_theta = np.exp(xbar)

# Delta method distribution
se_hat = (np.exp(xbar) * data.std()) / np.sqrt(n)
xs = np.linspace(*xlimits, 500)
ys = stats.norm.pdf(xs, loc=est_theta, scale=se_hat)
z = stats.norm.ppf(1 - alpha / 2)

delta_ci = est_theta + z * se_hat * np.array([-1, 1])
print(f"Delta method se: {se_hat:.4}")
print(f"Delta method 95% CI: {delta_ci}")

axs[0, 1].plot(xs, ys)
axs[0, 1].set(
    xlabel="Theta-hat", ylabel="Density", title="Delta method distribution"
)


# nonparametric bootstrap distribution
boots_data = rng.choice(data, size=(n, b))
nonparam_boots_thetas = np.exp(boots_data.mean(axis=0))

nonparam_se = nonparam_boots_thetas.std()
nonparam_ci = np.quantile(nonparam_boots_thetas, (alpha / 2, 1 - 0.5 * alpha))
print(f"Nonparametric bootstrap method se: {nonparam_se:.4}")
print(f"Nonparametric bootstrap method 95% CI: {nonparam_ci}")

axs[1, 0].hist(nonparam_boots_thetas, density=True, range=xlimits, bins=70)
axs[1, 0].set(
    xlabel="Theta-hat",
    ylabel="Density",
    title="Nonparametric bootstrap distribution",
)


# parametric bootstrap distribution
bootstrap_data = rng.normal(xbar, size=(n, b))
param_bootstrap_thetas = np.exp(bootstrap_data.mean(axis=0))

param_se = param_bootstrap_thetas.std()
param_ci = np.quantile(param_bootstrap_thetas, (alpha / 2, 1 - alpha / 2))
print(f"Parametric bootstrap method se: {param_se:.4}")
print(f"Parametric bootstrap method 95% CI: {param_ci}")

axs[1, 1].hist(param_bootstrap_thetas, density=True, range=xlimits, bins=70)
axs[1, 1].set(
    xlabel="Theta-hat",
    ylabel="Density",
    title="Parametric bootstrap distribution",
)


fig.savefig("09-09.png", bbox_inches="tight")
