import numpy as np
import scipy.stats as stats

rng = np.random.default_rng(42)

data_str = """
3.23 -2.50  1.88 -0.68  4.43 0.17
1.03 -0.07 -0.01  0.76  1.76 3.18
0.33 -0.31  0.30 -0.61  1.52 5.43
1.54  2.28  0.42  2.33 -1.03 4.00
0.39
"""
data = np.array([float(x) for x in data_str.split()])
n = len(data)

z = stats.norm.ppf(0.95)
mu_hat = data.mean()
sigma_hat = data.std()
tau_hat = z * sigma_hat + mu_hat

b = 5000
bootstrap_data = rng.normal(loc=mu_hat, scale=sigma_hat, size=(n, b))
tau_bootstrap = z * bootstrap_data.std(axis=0) + bootstrap_data.mean(axis=0)

bootstrap_se = tau_bootstrap.std()
delta_se = (1.534 * sigma_hat) / np.sqrt(n)

print(f"Standard error using the delta method: {delta_se:.4}")
print(f"Standard error using parametric bootstrap: {bootstrap_se:.4}")
