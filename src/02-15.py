import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def draw_exponential_samples(beta=1.0, size=10_000, seed=42):
    rng = np.random.default_rng(seed)
    uniform_x = rng.uniform(size=size)
    transformed_x = stats.expon.ppf(uniform_x, scale=1 / beta)
    return transformed_x


samples = draw_exponential_samples(beta=1)
plt.ylabel("Frequency")
plt.xlabel("x")
plt.hist(samples, bins=40)
plt.show()
