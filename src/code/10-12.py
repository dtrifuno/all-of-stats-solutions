import numpy as np
import scipy.stats as stats

rng = np.random.default_rng(42)

lambda_0 = 1
n = 20
alpha = 0.05
z_alpha = stats.norm.ppf(1 - alpha / 2)
sims = 200000
data = rng.poisson(lambda_0, size=(n, sims))
lambda_hats = data.mean(axis=0)
ws = np.sqrt(n) * np.abs(lambda_hats - lambda_0) / np.sqrt(lambda_hats)
test_outcomes = ws > z_alpha
type_i_rate = np.sum(test_outcomes) / sims

print(f"The type I error rate in {sims} simulations is {type_i_rate:.4}.")
