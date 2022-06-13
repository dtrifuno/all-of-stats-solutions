import numpy as np


def generate_multinomial(nsims, n, p, rng=np.random.default_rng()):
    p_cum = p.cumsum()
    k = p.size
    result = np.zeros(shape=(nsims, k))
    us = rng.uniform(size=(nsims, n))
    for i, u_row in enumerate(us):
        for u in u_row:
            j = np.argmax(u <= p_cum)
            result[i][j] += 1
    return result
