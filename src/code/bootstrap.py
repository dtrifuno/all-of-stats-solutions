# bootstrap.py

import numpy as np
import scipy.stats as stats


def create_normal_ci(t_hat, bootstrap_ts, alpha=0.05):
    se = bootstrap_ts.std()
    z = stats.norm.ppf(1 - alpha / 2)
    return (t_hat - z * se, t_hat + z * se)


def create_pivotal_ci(t_hat, bootstrap_ts, alpha=0.05):
    l = np.quantile(bootstrap_ts, alpha / 2)
    u = np.quantile(bootstrap_ts, 1 - alpha / 2)
    return (2 * t_hat - u, 2 * t_hat - l)


def create_percentile_ci(bootstrap_ts, alpha=0.05):
    l = np.quantile(bootstrap_ts, alpha / 2)
    u = np.quantile(bootstrap_ts, 1 - alpha / 2)
    return (l, u)


def between(x, ci):
    return x >= ci[0] and x <= ci[1]
