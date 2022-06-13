import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from bootstrap import create_percentile_ci

rng = np.random.default_rng(42)
np.set_printoptions(precision=3)

b = 10000
n = 100
data = rng.normal(5, size=n)
est_theta = np.exp(data.mean())

bootstrap_data = rng.choice(data, size=(n, b), replace=True)
bootstrap_thetas = np.exp(bootstrap_data.mean(axis=0))

est_se_theta = bootstrap_thetas.std()
percent_ci = create_percentile_ci(bootstrap_thetas)
print(f"The estimate for theta is {est_theta:.4}.")
print(f"The estimated se for theta is {est_se_theta:.4}.")
print(f"A 95% CI for theta is {percent_ci}.")

x_min = 0.1
x_max = 600
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
ax1.hist(bootstrap_thetas, bins="auto", density=True, range=(x_min, x_max))
ax1.set(xlabel="Bootstrapped theta-hat", ylabel="Density")

xs = np.linspace(x_min, x_max, 1000)
ys = stats.lognorm.pdf(xs, s=1, scale=np.exp(5 + 1 / (2 * n)))
ax2.plot(xs, ys)
ax2.set(xlabel="Theta-hat", ylabel="Density")
fig.savefig("08-06.png", bbox_inches="tight")
