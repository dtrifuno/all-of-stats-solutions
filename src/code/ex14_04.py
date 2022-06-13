import numpy as np
from scipy.linalg import sqrtm


def generate_box_muller(nsims, rng=np.random.default_rng()):
    """Generate standard normal variates using the Box-Muller transform"""
    s = nsims // 2 + 1
    us = rng.uniform(size=(2, s))
    z1 = np.sqrt(-2 * np.log(us[0])) * np.cos(2 * np.pi * us[1])
    z2 = np.sqrt(-2 * np.log(us[0])) * np.sin(2 * np.pi * us[1])
    zs = np.concatenate((z1, z2))
    return zs[:nsims]


def generate_multivariate_normal(nsims, mean, var, rng=np.random.default_rng()):
    n = mean.shape[0]
    zs = generate_box_muller(nsims * n, rng)
    zs_shaped = np.reshape(zs, (-1, n))
    sigma = sqrtm(var)
    return np.apply_along_axis(lambda v: sigma @ v + mean, 1, zs_shaped)
