import numpy as np
import scipy.stats as stats


def create_ecdf(xs):
    """Since numpy and scipy do not include a function for
    computing an empirical CDF, we write our own."""
    augmented_xs = np.append(xs, float("+inf"))
    sorted_xs = np.sort(augmented_xs)
    n = len(sorted_xs)
    ys = np.arange(0, n) / (n - 1)
    ecdf = np.vectorize(lambda x: ys[np.argmax(sorted_xs > x)])
    return ecdf


rng = np.random.default_rng(42)

sample_size = 100
alpha = 0.05
attempts = 1000
epsilon_n = np.sqrt(1 / (2 * sample_size) * np.log(2 / alpha))
xs = np.linspace(-14, 14, 1000)

for name, dist in [
    ("Standard normal", stats.norm),
    ("Standard Cauchy", stats.cauchy),
]:
    covered = 0
    for _ in range(attempts):
        samples = dist.rvs(size=sample_size, random_state=rng)
        ecdf = create_ecdf(samples)
        l = np.maximum(ecdf(xs) - epsilon_n, 0)
        u = np.minimum(ecdf(xs) + epsilon_n, 1)
        ys_true = dist.cdf(xs)
        if (l <= ys_true).all() & (ys_true <= u).all():
            covered += 1
    print(f"{name} coverage: {covered / attempts}")
