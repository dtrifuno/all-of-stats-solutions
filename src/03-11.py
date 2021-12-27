import matplotlib.pyplot as plt
import numpy as np


def simulate_stock_run(n=10_000, seed=42):
    rng = np.random.default_rng(seed)
    y = rng.choice([-1, 1], size=n)
    x = y.cumsum()
    plt.ylabel("Cumulative change in stock price", size=14)
    plt.xlabel("Day", size=14)
    plt.plot(x)
    plt.show()


simulate_stock_run(seed=42)
simulate_stock_run(seed=43)
simulate_stock_run(seed=44)
