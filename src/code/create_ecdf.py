import numpy as np


def create_ecdf(xs):
    """Since numpy and scipy do not include a function for
    computing an empirical CDF, we have to write our own."""
    augmented_xs = np.append(xs, float("+inf"))
    sorted_xs = np.sort(augmented_xs)
    n = len(sorted_xs)
    ys = np.arange(0, n) / (n - 1)
    ecdf = np.vectorize(lambda x: ys[np.argmax(sorted_xs > x)])
    return ecdf
