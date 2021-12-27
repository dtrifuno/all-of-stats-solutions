import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

# Check Exercise 7.3 for definition of create_ecdf
from create_ecdf import create_ecdf

df = pd.read_table("src/data/fijiquakes.dat", sep="\s+", index_col=0)
magnitudes = df["mag"]
alpha = 0.05
epsilon_n = np.sqrt(1 / (2 * magnitudes.size) * np.log(2 / alpha))

ecdf = create_ecdf(magnitudes)
xs = np.linspace(magnitudes.min() - 0.1, magnitudes.max() + 0.1, 200)
l = np.maximum(ecdf(xs) - epsilon_n, 0)
u = np.minimum(ecdf(xs) + epsilon_n, 1)

plt.plot(xs, ecdf(xs), label="empirical CDF")
plt.plot(xs, l, "--", label="0.95 lower bound")
plt.plot(xs, u, "--", label="0.95 upper bound")
plt.ylabel("Cumulative probability")
plt.xlabel("Earthquake magnitude")
plt.legend()
plt.show()

# We compute the confidence interval using Exercise 7.6
b = 4.9
a = 4.3
theta_hat = ecdf(b) - ecdf(a)
theta_se = np.sqrt(theta_hat * (1 - theta_hat) / magnitudes.size)
z = stats.norm.ppf(1 - alpha / 2)
l = theta_hat - z * theta_se
u = theta_hat + z * theta_se

print(f"F({b}) - F({a}) is approximately {theta_hat}.")
print(f"A 95% CI for this value is given by [{l:.4}, {u:.4}].")
