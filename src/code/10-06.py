import numpy as np
import scipy.stats as stats

x = 922
n = 1919
p_hat = x / n
p_0 = 1 / 2
se_hat = np.sqrt(p_hat * (1 - p_hat) / n)
w = np.abs(p_hat - p_0) / se_hat
p_value = 2 * stats.norm.cdf(-w)
print(f"The p-value is {p_value:.4}.")

z_alpha = stats.norm.ppf(1 - 0.05 / 2)
ci = (p_hat - z_alpha * se_hat, p_hat + z_alpha * se_hat)
print(f"A 95% CI for p is [{ci[0]:.4}, {ci[1]:.4}].")
