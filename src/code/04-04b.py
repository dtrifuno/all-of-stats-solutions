import matplotlib.pyplot as plt
import numpy as np

p = 0.4
alpha = 0.05
coverages = []
rng = np.random.default_rng(42)
ns = np.arange(1, 10000, 50)
for n in ns:
    epsilon_n = np.sqrt((1 / (2 * n)) * np.log(2 / alpha))
    p_hats = rng.binomial(n, p, size=50000) / n
    satisfy_both = (p_hats + epsilon_n >= p) & (p_hats - epsilon_n <= p)
    coverages.append(np.mean(satisfy_both))

plt.scatter(ns, coverages)
plt.xlabel("n")
plt.ylabel("Coverage")
plt.savefig("04-04b.png", bbox_inches="tight")
