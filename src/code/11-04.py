import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# (a)
n_1 = 50
n_2 = 50
x_1 = 40
x_2 = 30
p_hat_1 = x_1 / n_1
p_hat_2 = x_2 / n_2
tau_hat = p_hat_1 - p_hat_2
tau_se_hat = np.sqrt(
    p_hat_1 * (1 - p_hat_1) / n_1 + p_hat_2 * (1 - p_hat_2) / n_2
)
alpha = 0.1
z_alpha = stats.norm.ppf(1 - alpha / 2)
ci = (tau_hat - z_alpha * tau_se_hat, tau_hat + z_alpha * tau_se_hat)
print(f"MLE for tau is {tau_hat:.4}.")
print(f"Standard error for tau is {tau_se_hat:.4}.")
print(f"90% CI for tau is [{ci[0]:.4}, {ci[1]:.4}].")
print()

# (b)
sims = 5_000
rng = np.random.default_rng(42)
x_1_draws = rng.binomial(n_1, p_hat_1, size=sims)
x_2_draws = rng.binomial(n_2, p_hat_2, size=sims)
tau_draws = x_1_draws / n_1 - x_2_draws / n_2
bootstrap_ci = (
    np.quantile(tau_draws, alpha / 2),
    np.quantile(tau_draws, 1 - alpha / 2),
)
print(
    "90% parametric bootstrap CI for tau is",
    f"[{bootstrap_ci[0]:.4}, {bootstrap_ci[1]:.4}].",
)
print()

# (c)
taus = np.arange(-n_2, n_1 + 1)
p_1_posterior_draws = rng.beta(x_1 + 1, n_1 - x_1 + 1, size=sims)
p_2_posterior_draws = rng.beta(x_2 + 1, n_2 - x_2 + 1, size=sims)
tau_posterior_draws = p_1_posterior_draws - p_2_posterior_draws
tau_posterior_mean = tau_posterior_draws.mean()
posterior_ci = (
    np.quantile(tau_posterior_draws, alpha / 2),
    np.quantile(tau_posterior_draws, 1 - alpha / 2),
)
print(f"The posterior mean for tau is {tau_posterior_mean:.4}.")
print(
    "90% posterior CI for tau is",
    f"[{posterior_ci[0]:.4}, {posterior_ci[1]:.4}].",
)
print()

# (d)
psi_hat = np.log((p_hat_1 / (1 - p_hat_1)) / (p_hat_2 / (1 - p_hat_2)))
psi_se_hat = np.sqrt(
    1 / (n_1 * (p_hat_1 - p_hat_1 ** 2)) + 1 / (n_2 * (p_hat_2 ** 2 - p_hat_2))
)
psi_ci = (psi_hat - z_alpha * psi_se_hat, psi_hat + z_alpha * psi_se_hat)
print(f"MLE for psi is {psi_hat:.4}.")
print(f"Standard error for psi is {psi_se_hat:.4}.")
print(f"90% CI for psi is [{psi_ci[0]:.4}, {psi_ci[1]:.4}].")
print()


# (e)
psi_posterior_draws = np.log(
    (p_1_posterior_draws / (1 - p_1_posterior_draws))
    / (p_2_posterior_draws / (1 - p_2_posterior_draws))
)
psi_posterior_mean = psi_posterior_draws.mean()
psi_posterior_ci = (
    np.quantile(psi_posterior_draws, alpha / 2),
    np.quantile(psi_posterior_draws, 1 - alpha / 2),
)
print(f"The posterior mean for psi is {psi_posterior_mean:.4}.")
print(
    "90% posterior CI for psi is",
    f"[{psi_posterior_ci[0]:.4}, {psi_posterior_ci[1]:.4}].",
)
