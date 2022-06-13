# bootstrap.py

import numpy as np
import scipy.stats as stats


def create_normal_ci(t_hat, bootstrap_ts, alpha=0.05):
    se = bootstrap_ts.std()
    z = stats.norm.ppf(1 - alpha / 2)
    return t_hat + np.array([-1, 1]) * z * se


def create_pivotal_ci(t_hat, bootstrap_ts, alpha=0.05):
    qs = np.quantile(bootstrap_ts, (1 - alpha / 2, alpha / 2))
    return 2 * t_hat - qs


def create_percentile_ci(bootstrap_ts, alpha=0.05):
    return np.quantile(bootstrap_ts, (alpha / 2, 1 - alpha / 2))


def between(x, ci):
    return x >= ci[0] and x <= ci[1]
