import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)

n = 50
b = 10000
data = rng.uniform(0, 1, size=n)
est_theta = data.max()

bootstrap_data = rng.choice(data, size=(n, b), replace=True)
bootstrap_thetas = bootstrap_data.max(axis=0)

x_min = 0.01
x_max = 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
ax1.hist(bootstrap_thetas, bins=100, density=True, range=(x_min, x_max))
ax1.set(xlabel="Bootstrapped theta-hat", ylabel="Density")

xs = np.linspace(x_min, x_max, 1000)
ys = 50 * (xs ** 49)
ax2.plot(xs, ys)
ax2.set(xlabel="Theta-hat", ylabel="Density")
fig.savefig("08-07.png", bbox_inches="tight")
